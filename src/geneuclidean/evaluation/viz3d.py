"""Studio-Grade Dual-Mode 3D Molecular & Protein-Ligand Visualization Suite.

Provides two complementary, zero-external-GUI 3D visualization engines for GenEuclidean v3.1:

- **Mode A — Interactive Browser WebGL Studio HTML**
  (:func:`render_interactive_3d_html`, :func:`render_interactive_3d_gallery_html`):
  Self-contained dark-mode scientific HTML5/WebGL apps powered by ``3Dmol.js`` with
  inline embedded PDB, SDF, 2D SVG structure, 3D Perceiver attention maps, and exact
  non-covalent interaction cylinders. Features toggleable translucent molecular surface,
  chain / hydrophobicity / 3D attention coloring, native co-crystal ghost overlay,
  per-channel interaction toggles, interactive zoom-on-click residue table, camera
  presets, turntable spin, PNG snapshot export, and multi-target gallery navigation.

- **Mode B — 300-DPI Publication Static 3D Renders**
  (:func:`render_static_3d_complex_png`, :func:`render_static_3d_molecule_png`):
  Pure-Python Matplotlib (``Agg`` backend) custom perspective-projected, painter's-algorithm
  depth-sorted, pseudo-Phong specular-shaded 3D molecular renderer supporting both
  *dark obsidian* (``#0B1120``) and *crisp publication white* (``#FFFFFF``) themes.
  Automatically computes an occlusion-free PCA camera orientation, draws multi-order
  cylindrical bonds (single, double, triple, aromatic), protein backbone spline trace,
  translucent attention-colored pocket cavity glow, interacting side-chain sticks,
  dashed 3D non-covalent interaction vectors with de-overlapped residue callouts, and
  an accompanying 2D RDKit structure + medicinal chemistry KPI side panel.
"""

from __future__ import annotations

import contextlib
import html as html_lib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from geneuclidean.evaluation.interactions3d import (
    ALL_INTERACTION_KINDS,
    InteractionProfile,
    InteractionRecord,
    profile_3d_interactions,
    profile_3d_interactions_from_pdb_text,
)

# RDKit populates many C++ Boost.Python descriptor & conformer helpers dynamically at import time.
_ALLCHEM: Any = AllChem
_DESCRIPTORS: Any = Descriptors

# ---------------------------------------------------------------------------
# Theme, CPK & Interaction Visual Palettes
# ---------------------------------------------------------------------------

ThemeName = Literal["dark", "light"]

_DPI = 300

_INTERACTION_META: dict[str, dict[str, str]] = {
    "hbond": {
        "label": "H-Bond",
        "short": "HB",
        "dark": "#38BDF8",  # Electric cyan
        "light": "#0284C7",  # Rich sky-cobalt
    },
    "salt_bridge": {
        "label": "Salt Bridge",
        "short": "SB",
        "dark": "#F472B6",  # Neon magenta
        "light": "#DB2777",  # Deep magenta
    },
    "pi_stacking": {
        "label": "π–π Stacking",
        "short": "π–π",
        "dark": "#FBBF24",  # Luminous gold
        "light": "#D97706",  # Warm amber
    },
    "hydrophobic": {
        "label": "Hydrophobic",
        "short": "HYD",
        "dark": "#34D399",  # Vivid emerald
        "light": "#059669",  # Forest emerald
    },
    "cation_pi": {
        "label": "Cation–π",
        "short": "CAT-π",
        "dark": "#FB923C",  # Tangerine
        "light": "#EA580C",  # Burnt orange
    },
    "halogen_bond": {
        "label": "Halogen Bond",
        "short": "XB",
        "dark": "#A78BFA",  # Violet
        "light": "#7C3AED",  # Royal violet
    },
}

_CPK_DARK: dict[str, str] = {
    "C": "#22D3EE",  # Studio cyan-teal carbon (vivid against obsidian)
    "N": "#3B82F6",  # Cobalt blue
    "O": "#EF4444",  # Crimson red
    "S": "#FACC15",  # Canary yellow
    "P": "#F97316",  # Orange
    "F": "#4ADE80",  # Bright mint green
    "CL": "#22C55E",  # Vivid green
    "BR": "#A855F7",  # Amethyst purple
    "I": "#C084FC",  # Soft violet
    "H": "#E2E8F0",  # Cool silver-white
    "B": "#FB7185",
    "SI": "#94A3B8",
}

_CPK_LIGHT: dict[str, str] = {
    "C": "#334155",  # Crisp slate-graphite carbon (high contrast on white)
    "N": "#2563EB",  # Deep cobalt
    "O": "#DC2626",  # Publication red
    "S": "#CA8A04",  # Ochre gold
    "P": "#EA580C",  # Burnt orange
    "F": "#16A34A",  # Emerald green
    "CL": "#15803D",  # Deep green
    "BR": "#7E22CE",  # Deep purple
    "I": "#6B21A8",  # Plum
    "H": "#CBD5E1",  # Light slate
    "B": "#E11D48",
    "SI": "#64748B",
}

# Ball-and-stick visual sphere radii (Å) proportional to covalent radii
_ATOM_VISUAL_RADIUS: dict[str, float] = {
    "H": 0.21,
    "C": 0.36,
    "N": 0.35,
    "O": 0.34,
    "F": 0.32,
    "P": 0.44,
    "S": 0.43,
    "CL": 0.42,
    "BR": 0.46,
    "I": 0.50,
    "B": 0.38,
    "SI": 0.44,
}

# Kyte-Doolittle residue hydropathy normalized into [-1, 1] for WebGL coloring
_WATER_RESNAMES: frozenset[str] = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL"})


# ---------------------------------------------------------------------------
# Public Spec Dataclass for Gallery & Single Complex Rendering
# ---------------------------------------------------------------------------


@dataclass
class ComplexShowcaseSpec:
    """Declarative specification for a 3D protein-ligand complex showcase card.

    Used by both :func:`render_interactive_3d_html` (single view) and
    :func:`render_interactive_3d_gallery_html` (multi-target dashboard).
    """

    target_id: str
    title: str
    pdb_id: str = ""
    disease: str = ""
    pocket_pdb_path: str | Path | None = None
    pocket_pdb_text: str | None = None
    ligand_mol: Chem.Mol | None = None
    native_ligand_mol: Chem.Mol | None = None
    smiles: str = ""
    interactions: InteractionProfile | None = None
    residue_attention: Mapping[str, float] | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Color & Gradient Helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> NDArray[np.float64]:
    """Convert ``#RRGGBB`` hex color string to float RGB array in ``[0, 1]``."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return np.array(
        [int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0],
        dtype=np.float64,
    )


def _rgb_to_hex(rgb: NDArray[np.float64]) -> str:
    """Convert float RGB array in ``[0, 1]`` to ``#RRGGBB`` hex string."""
    clipped = np.clip(np.round(rgb * 255.0), 0, 255).astype(int)
    return f"#{clipped[0]:02X}{clipped[1]:02X}{clipped[2]:02X}"


def _blend_rgb(
    c1: NDArray[np.float64],
    c2: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Linearly interpolate between two RGB colors by ``t in [0, 1]``."""
    alpha = max(0.0, min(1.0, float(t)))
    return (1.0 - alpha) * c1 + alpha * c2


def _attention_color_rgb(score: float, theme: ThemeName = "dark") -> NDArray[np.float64]:
    """Map a normalized 3D Perceiver attention weight ``[0, 1]`` to indigo → amber → gold."""
    val = max(0.0, min(1.0, float(score)))
    if theme == "dark":
        stops = [
            (0.00, _hex_to_rgb("#1E1B4B")),  # Deep obsidian indigo
            (0.35, _hex_to_rgb("#4F46E5")),  # Electric indigo
            (0.70, _hex_to_rgb("#F59E0B")),  # Luminous amber
            (1.00, _hex_to_rgb("#FDE047")),  # Radiant gold
        ]
    else:
        stops = [
            (0.00, _hex_to_rgb("#CBD5E1")),  # Muted cool slate
            (0.35, _hex_to_rgb("#6366F1")),  # Indigo
            (0.70, _hex_to_rgb("#EA580C")),  # Rich orange-amber
            (1.00, _hex_to_rgb("#B45309")),  # Deep gold-bronze
        ]
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:], strict=True):
        if val <= p1:
            t = (val - p0) / max(p1 - p0, 1e-8)
            return _blend_rgb(c0, c1, t)
    return stops[-1][1]


def _lookup_residue_attention(
    residue_id: str,
    resseq: int,
    attention_map: Mapping[str, float] | None,
) -> float | None:
    """Resolve residue attention score from flexible keys (``ASP103:A``, ``ASP103``, ``103``)."""
    if not attention_map:
        return None
    if residue_id in attention_map:
        return float(attention_map[residue_id])
    base = residue_id.split(":")[0]
    if base in attention_map:
        return float(attention_map[base])
    seq_str = str(resseq)
    if seq_str in attention_map:
        return float(attention_map[seq_str])
    return None


def _short_residue_label(residue_id: str) -> str:
    """Shorten ``ASP103:A`` to ``ASP103`` on chain A, keeping explicit non-A chain labels."""
    parts = residue_id.split(":")
    if len(parts) == 2 and parts[1] in {"", "A"}:
        return parts[0]
    return residue_id


def _element_color(
    element: str,
    theme: ThemeName,
    *,
    carbon_override: str | None = None,
) -> NDArray[np.float64]:
    """Return studio CPK RGB color for an element symbol under the active theme."""
    el = element.strip().upper()
    if el == "C" and carbon_override is not None:
        return _hex_to_rgb(carbon_override)
    palette = _CPK_DARK if theme == "dark" else _CPK_LIGHT
    fallback = "#94A3B8" if theme == "dark" else "#475569"
    return _hex_to_rgb(palette.get(el, fallback))


# ---------------------------------------------------------------------------
# Pocket Structural Geometry Extraction (for Static 3D Rendering)
# ---------------------------------------------------------------------------


@dataclass
class _PocketAtom3D:
    """Lightweight pocket heavy atom record for 3D projection and bond rendering."""

    residue_id: str
    resname: str
    resseq: int
    chain_id: str
    atom_name: str
    element: str
    xyz: NDArray[np.float64]


@dataclass
class _PocketGeometry:
    """Parsed 3D protein pocket coordinates grouped by chain and residue."""

    atoms: list[_PocketAtom3D]
    by_residue: dict[str, list[_PocketAtom3D]]
    ca_traces: list[list[tuple[str, int, NDArray[np.float64]]]]
    aromatic_polygons: dict[str, list[NDArray[np.float64]]]


_RING_NAME_SPECS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    ),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HID": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIE": (("CG", "ND1", "CE1", "NE2", "CD2"),),
    "HIP": (("CG", "ND1", "CE1", "NE2", "CD2"),),
}


def _load_pocket_geometry(
    pocket_pdb_path: str | Path | None = None,
    pocket_pdb_text: str | None = None,
) -> _PocketGeometry:
    """Extract heavy atoms, contiguous Cα traces, and aromatic ring polygons from a pocket PDB."""
    parser = PDBParser(QUIET=True)
    if pocket_pdb_text is not None:
        structure = parser.get_structure("pocket", io.StringIO(pocket_pdb_text))
    elif pocket_pdb_path is not None:
        structure = parser.get_structure("pocket", str(pocket_pdb_path))
    else:
        return _PocketGeometry(atoms=[], by_residue={}, ca_traces=[], aromatic_polygons={})

    models = list(structure.get_models())
    if not models:
        return _PocketGeometry(atoms=[], by_residue={}, ca_traces=[], aromatic_polygons={})

    atoms: list[_PocketAtom3D] = []
    by_residue: dict[str, list[_PocketAtom3D]] = {}
    ca_traces: list[list[tuple[str, int, NDArray[np.float64]]]] = []
    aromatic_polygons: dict[str, list[NDArray[np.float64]]] = {}

    for chain in models[0]:
        chain_id = str(chain.id).strip() or "A"
        current_trace: list[tuple[str, int, NDArray[np.float64]]] = []

        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if resname in _WATER_RESNAMES or (residue.id[0] != " " and resname not in {"MSE"}):
                continue
            _, resseq, icode = residue.id
            icode_clean = icode.strip() if isinstance(icode, str) else ""
            res_id = f"{resname}{resseq}{icode_clean}:{chain_id}"

            res_atoms: list[_PocketAtom3D] = []
            atom_by_name: dict[str, NDArray[np.float64]] = {}
            for atom in residue.get_atoms():
                name = atom.get_name().strip().upper()
                raw_el = str(getattr(atom, "element", "")).strip().upper()
                el = raw_el if raw_el else "".join(c for c in name if c.isalpha())[:1] or "C"
                if el in {"H", "D"}:
                    continue
                xyz = np.asarray(atom.get_coord(), dtype=np.float64)
                rec = _PocketAtom3D(
                    residue_id=res_id,
                    resname=resname,
                    resseq=int(resseq),
                    chain_id=chain_id,
                    atom_name=name,
                    element=el,
                    xyz=xyz,
                )
                atoms.append(rec)
                res_atoms.append(rec)
                atom_by_name[name] = xyz

            if not res_atoms:
                continue
            by_residue[res_id] = res_atoms

            # Track contiguous Cα backbone segments (chain breaks > 4.6 Å start a new ribbon)
            if "CA" in atom_by_name:
                ca_xyz = atom_by_name["CA"]
                if current_trace and float(np.linalg.norm(ca_xyz - current_trace[-1][2])) > 4.6:
                    if len(current_trace) >= 2:
                        ca_traces.append(current_trace)
                    current_trace = []
                current_trace.append((res_id, int(resseq), ca_xyz))

            # Ordered aromatic ring polygons for clean translucent aromatic fill
            for ring_names in _RING_NAME_SPECS.get(resname, ()):
                if all(n in atom_by_name for n in ring_names):
                    poly = np.stack([atom_by_name[n] for n in ring_names], axis=0)
                    aromatic_polygons.setdefault(res_id, []).append(poly)

        if len(current_trace) >= 2:
            ca_traces.append(current_trace)

    return _PocketGeometry(
        atoms=atoms,
        by_residue=by_residue,
        ca_traces=ca_traces,
        aromatic_polygons=aromatic_polygons,
    )


def _catmull_rom_spline(
    points: NDArray[np.float64],
    subdivisions: int = 7,
) -> NDArray[np.float64]:
    """Interpolate a smooth 3D Catmull-Rom backbone spline through Cα control points."""
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        ts = np.linspace(0.0, 1.0, subdivisions + 1)[:, None]
        return np.asarray((1.0 - ts) * points[0] + ts * points[1], dtype=np.float64)

    # Extrapolate phantom endpoints for C1 continuity at ribbon ends
    p_ext = np.vstack([2.0 * points[0] - points[1], points, 2.0 * points[-1] - points[-2]])
    out: list[NDArray[np.float64]] = []
    for i in range(1, n):
        p0, p1, p2, p3 = p_ext[i - 1], p_ext[i], p_ext[i + 1], p_ext[i + 2]
        t_vals = np.linspace(0.0, 1.0, subdivisions, endpoint=(i == n - 1))
        for t in t_vals:
            t2 = t * t
            t3 = t2 * t
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Optimal PCA Camera & Perspective Projection Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Camera3D:
    """Right-handed orthonormal 3D camera with gentle perspective projection."""

    origin: NDArray[np.float64]
    basis: NDArray[np.float64]  # shape (3, 3), rows = [right, up, forward_to_viewer]
    focal_dist: float

    def transform(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map world 3D points ``(..., 3)`` into camera coordinates ``(x, y, z)``."""
        return (pts - self.origin) @ self.basis.T

    def project(self, pts: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project world 3D points ``(N, 3)`` to screen ``(N, 2)`` and camera depth ``z``."""
        cam = self.transform(pts)
        denom = np.maximum(self.focal_dist - cam[:, 2], 2.5)
        scale = self.focal_dist / denom
        xy = cam[:, :2] * scale[:, None]
        return xy, cam[:, 2]

    def project_point(self, pt: NDArray[np.float64]) -> tuple[float, float, float, float]:
        """Project a single 3D point returning ``(x2d, y2d, z_cam, persp_scale)``."""
        cam = (pt - self.origin) @ self.basis.T
        denom = max(self.focal_dist - float(cam[2]), 2.5)
        scale = self.focal_dist / denom
        return float(cam[0]) * scale, float(cam[1]) * scale, float(cam[2]), float(scale)


def _rotation_matrix_axis_angle(axis: NDArray[np.float64], angle_deg: float) -> NDArray[np.float64]:
    """Rodrigues 3D rotation matrix around unit ``axis`` by ``angle_deg`` degrees."""
    if abs(angle_deg) < 1e-9:
        return np.eye(3, dtype=np.float64)
    rad = math.radians(angle_deg)
    k = axis / max(float(np.linalg.norm(axis)), 1e-8)
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64) + math.sin(rad) * skew + (1.0 - math.cos(rad)) * (skew @ skew)
    )


def _compute_optimal_camera(
    ligand_coords: NDArray[np.float64],
    *,
    pocket_coords: NDArray[np.float64] | None = None,
    contact_coords: NDArray[np.float64] | None = None,
    elev_deg: float = 11.0,
    azim_deg: float = 9.0,
) -> _Camera3D:
    """Compute an occlusion-minimizing PCA camera framing the ligand & pocket opening.

    Places the ligand's largest-variance axis horizontally, second-largest vertically,
    and views along the smallest-variance axis (opening the pocket cavity toward the
    camera when pocket coordinates are provided).
    """
    origin = np.mean(ligand_coords, axis=0)

    # Blend ligand heavy atoms with interaction midpoints so contacts lie open in the viewing plane
    centered_parts = [ligand_coords - origin]
    if contact_coords is not None and len(contact_coords) > 0:
        centered_parts.append(0.55 * (contact_coords - origin))
    pts = np.vstack(centered_parts)

    if len(pts) >= 3:
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        right = vh[0]
        up = vh[1]
        forward = np.cross(right, up)
        f_norm = float(np.linalg.norm(forward))
        if f_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / f_norm
            up = np.cross(forward, right)
            up = up / max(float(np.linalg.norm(up)), 1e-8)
    else:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Flip viewing direction if nearby pocket atoms sit in front of the ligand
    if pocket_coords is not None and len(pocket_coords) > 0:
        dists = np.linalg.norm(pocket_coords - origin, axis=1)
        near = pocket_coords[dists <= 9.0]
        if len(near) == 0:
            near = pocket_coords
        mean_pocket_z = float(np.mean((near - origin) @ forward))
        if mean_pocket_z > 0.0:
            forward = -forward
            right = -right

    basis = np.vstack([right, up, forward])

    # Apply subtle studio viewing angle so flat aromatic planes reveal 3D depth
    if abs(azim_deg) > 1e-6:
        r_azim = _rotation_matrix_axis_angle(basis[1], azim_deg)
        basis = basis @ r_azim.T
    if abs(elev_deg) > 1e-6:
        r_elev = _rotation_matrix_axis_angle(basis[0], elev_deg)
        basis = basis @ r_elev.T

    scene_radius = max(float(np.max(np.linalg.norm(ligand_coords - origin, axis=1))), 3.5)
    focal_dist = max(26.0, 4.4 * scene_radius)
    return _Camera3D(origin=origin, basis=basis, focal_dist=focal_dist)


# ---------------------------------------------------------------------------
# Pseudo-Phong Shaded 3D Primitives (Painter's Algorithm Depth-Sorted)
# ---------------------------------------------------------------------------


def _apply_depth_fog(
    rgb: NDArray[np.float64],
    bg_rgb: NDArray[np.float64],
    z_val: float,
    z_min: float,
    z_max: float,
    strength: float = 0.32,
) -> NDArray[np.float64]:
    """Blend color gently toward background as depth recedes for 3D atmospheric perspective."""
    span = max(z_max - z_min, 1.0)
    depth_t = max(0.0, min(1.0, (z_max - z_val) / span))
    return _blend_rgb(rgb, bg_rgb, depth_t * strength)


def _draw_phong_sphere(
    ax: Axes,
    x: float,
    y: float,
    radius: float,
    base_rgb: NDArray[np.float64],
    bg_rgb: NDArray[np.float64],
    *,
    zorder: float,
    alpha: float = 1.0,
    specular: float = 0.88,
) -> None:
    """Draw a glossy pseudo-Phong specular sphere via layered off-center shaded discs."""
    white = np.ones(3, dtype=np.float64)
    dark_rim = _blend_rgb(base_rgb, bg_rgb * 0.25, 0.58)
    shadow_body = _blend_rgb(base_rgb, np.zeros(3, dtype=np.float64), 0.24)
    lit_mid = _blend_rgb(base_rgb, white, 0.20)
    bright_mid = _blend_rgb(base_rgb, white, 0.54)
    glint = _blend_rgb(base_rgb, white, specular)

    # 1) Crisp rim silhouette
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            facecolor=tuple(dark_rim),
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
    )
    # 2) Ambient-shaded sphere core
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius * 0.92,
            facecolor=tuple(shadow_body),
            edgecolor="none",
            alpha=alpha,
            zorder=zorder + 0.0001,
        )
    )
    # 3) Key-lit diffuse dome (light coming from upper-left front)
    ax.add_patch(
        Circle(
            (x - 0.13 * radius, y + 0.14 * radius),
            radius=radius * 0.76,
            facecolor=tuple(lit_mid),
            edgecolor="none",
            alpha=alpha,
            zorder=zorder + 0.0002,
        )
    )
    # 4) Soft secondary specular bloom
    ax.add_patch(
        Circle(
            (x - 0.23 * radius, y + 0.25 * radius),
            radius=radius * 0.44,
            facecolor=tuple(bright_mid),
            edgecolor="none",
            alpha=alpha * 0.90,
            zorder=zorder + 0.0003,
        )
    )
    # 5) Crisp pinpoint specular highlight
    ax.add_patch(
        Circle(
            (x - 0.30 * radius, y + 0.32 * radius),
            radius=radius * 0.19,
            facecolor=tuple(glint),
            edgecolor="none",
            alpha=alpha * 0.92,
            zorder=zorder + 0.0004,
        )
    )


def _draw_shaded_cylinder_segment(
    ax: Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    rgb: NDArray[np.float64],
    bg_rgb: NDArray[np.float64],
    *,
    width_pt: float,
    zorder: float,
    dashed: bool = False,
    alpha: float = 1.0,
) -> None:
    """Draw a glossy shaded cylindrical bond segment with dark rim + specular axial highlight."""
    white = np.ones(3, dtype=np.float64)
    rim_col = _blend_rgb(rgb, bg_rgb * 0.2, 0.55)
    core_col = _blend_rgb(rgb, white, 0.08)
    spec_col = _blend_rgb(rgb, white, 0.58)

    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    linestyle = (0, (3.2, 2.2)) if dashed else "-"

    ax.plot(
        xs,
        ys,
        color=tuple(rim_col),
        lw=width_pt * 1.28,
        linestyle=linestyle,
        solid_capstyle="round",
        dash_capstyle="round",
        alpha=alpha,
        zorder=zorder,
    )
    ax.plot(
        xs,
        ys,
        color=tuple(core_col),
        lw=width_pt * 0.92,
        linestyle=linestyle,
        solid_capstyle="round",
        dash_capstyle="round",
        alpha=alpha,
        zorder=zorder + 0.0001,
    )
    if not dashed and width_pt >= 2.2:
        ax.plot(
            xs,
            ys,
            color=tuple(spec_col),
            lw=max(0.7, width_pt * 0.26),
            solid_capstyle="round",
            alpha=alpha * 0.65,
            zorder=zorder + 0.0002,
        )


def _draw_multi_order_bond(
    ax: Axes,
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    rgb1: NDArray[np.float64],
    rgb2: NDArray[np.float64],
    bg_rgb: NDArray[np.float64],
    *,
    bond_type: Chem.BondType,
    is_aromatic: bool,
    persp_scale: float,
    zorder: float,
    base_width_pt: float = 4.6,
    offset_world: float = 0.095,
) -> None:
    """Render a 2-tone cylindrical bond clearly distinguishing single/double/triple/aromatic."""
    dx = xy2[0] - xy1[0]
    dy = xy2[1] - xy1[1]
    length = math.hypot(dx, dy)
    if length < 1e-5:
        return

    # Perpendicular unit normal in screen coordinates for parallel multi-bond cylinders
    nx = -dy / length
    ny = dx / length
    off = offset_world * persp_scale

    # Spec list: (offset_multiplier, width_scale, is_dashed)
    if is_aromatic or bond_type == Chem.BondType.AROMATIC:
        cylinders = [(-0.70, 0.78, False), (0.85, 0.56, True)]
    elif bond_type == Chem.BondType.DOUBLE:
        cylinders = [(-0.82, 0.68, False), (0.82, 0.68, False)]
    elif bond_type == Chem.BondType.TRIPLE:
        cylinders = [(-1.25, 0.52, False), (0.0, 0.56, False), (1.25, 0.52, False)]
    else:
        cylinders = [(0.0, 1.0, False)]

    for mult, w_scale, is_dashed in cylinders:
        ox = nx * off * mult
        oy = ny * off * mult
        a = (xy1[0] + ox, xy1[1] + oy)
        b = (xy2[0] + ox, xy2[1] + oy)
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        w = max(1.4, base_width_pt * persp_scale * w_scale)

        # Slightly shorten dashed aromatic companion line inside the ring for clean corners
        if is_dashed:
            trim = 0.14
            a_trim = (a[0] + trim * (b[0] - a[0]), a[1] + trim * (b[1] - a[1]))
            b_trim = (b[0] - trim * (b[0] - a[0]), b[1] - trim * (b[1] - a[1]))
            mid_trim = (0.5 * (a_trim[0] + b_trim[0]), 0.5 * (a_trim[1] + b_trim[1]))
            _draw_shaded_cylinder_segment(
                ax, a_trim, mid_trim, rgb1, bg_rgb, width_pt=w, zorder=zorder, dashed=True
            )
            _draw_shaded_cylinder_segment(
                ax, mid_trim, b_trim, rgb2, bg_rgb, width_pt=w, zorder=zorder, dashed=True
            )
        else:
            _draw_shaded_cylinder_segment(
                ax, a, mid, rgb1, bg_rgb, width_pt=w, zorder=zorder, dashed=False
            )
            _draw_shaded_cylinder_segment(
                ax, mid, b, rgb2, bg_rgb, width_pt=w, zorder=zorder, dashed=False
            )


def _deoverlap_label_positions(
    anchors: list[tuple[float, float]],
    initial_positions: list[tuple[float, float]],
    *,
    box_w: float,
    box_h: float,
    bounds: tuple[float, float, float, float],
    iterations: int = 35,
) -> list[tuple[float, float]]:
    """Iteratively repel overlapping callout pills while keeping them near their anchors."""
    if len(initial_positions) <= 1:
        return list(initial_positions)

    xmin, xmax, ymin, ymax = bounds
    pos = np.array(initial_positions, dtype=np.float64)
    anc = np.array(anchors, dtype=np.float64)

    for _ in range(iterations):
        forces = np.zeros_like(pos)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                overlap_x = box_w - abs(dx)
                overlap_y = box_h - abs(dy)
                if overlap_x > 0.0 and overlap_y > 0.0:
                    # Push apart primarily along the axis of smallest overlap
                    if overlap_x < overlap_y:
                        sign = 1.0 if dx >= 0.0 else -1.0
                        push = 0.55 * overlap_x
                        forces[i, 0] += sign * push
                        forces[j, 0] -= sign * push
                    else:
                        sign = 1.0 if dy >= 0.0 else -1.0
                        push = 0.55 * overlap_y
                        forces[i, 1] += sign * push
                        forces[j, 1] -= sign * push

            # Gentle tether back toward anchor to avoid drifting away from the residue
            forces[i] += 0.03 * (anc[i] - pos[i])

        pos += forces
        pos[:, 0] = np.clip(pos[:, 0], xmin + 0.55 * box_w, xmax - 0.55 * box_w)
        pos[:, 1] = np.clip(pos[:, 1], ymin + 0.55 * box_h, ymax - 0.55 * box_h)

    return [(float(p[0]), float(p[1])) for p in pos]


# ---------------------------------------------------------------------------
# RDKit 2D Structure Renderers (Cairo PNG for Static Side Panel & SVG for WebGL)
# ---------------------------------------------------------------------------


def _prepare_2d_mol(mol: Chem.Mol) -> Chem.Mol:
    """Create a clean 2D hydrogen-stripped depiction copy of ``mol``."""
    mol_2d = Chem.RemoveHs(Chem.Mol(mol))
    _ALLCHEM.Compute2DCoords(mol_2d)
    with contextlib.suppress(Exception):
        rdMolDraw2D.PrepareMolForDrawing(mol_2d)
    return mol_2d


def _configure_rdkit_draw_options(
    opts: Any,
    theme: ThemeName,
) -> None:
    """Style RDKit 2D drawing options to match our dark obsidian or light studio theme."""
    opts.bondLineWidth = 2.4
    opts.addStereoAnnotation = True
    opts.padding = 0.12
    opts.fixedBondLength = 32.0
    if theme == "dark":
        opts.setBackgroundColour((15 / 255.0, 23 / 255.0, 42 / 255.0, 1.0))
        opts.setHighlightColour((56 / 255.0, 189 / 255.0, 248 / 255.0, 0.35))
        opts.updateAtomPalette(
            {
                0: (0.88, 0.92, 0.97, 1.0),
                6: (0.88, 0.92, 0.97, 1.0),  # Crisp silver-white carbons on dark card
                7: (0.38, 0.65, 0.98, 1.0),  # Bright cobalt N
                8: (0.97, 0.44, 0.44, 1.0),  # Luminous coral-red O
                9: (0.29, 0.87, 0.50, 1.0),  # Mint F
                15: (0.98, 0.57, 0.24, 1.0),  # Orange P
                16: (0.98, 0.80, 0.08, 1.0),  # Gold S
                17: (0.20, 0.83, 0.60, 1.0),  # Emerald Cl
                35: (0.75, 0.52, 0.99, 1.0),  # Lavender Br
                53: (0.85, 0.47, 0.94, 1.0),  # Violet I
            }
        )
    else:
        opts.setBackgroundColour((248 / 255.0, 250 / 255.0, 252 / 255.0, 1.0))
        opts.updateAtomPalette(
            {
                0: (0.12, 0.16, 0.23, 1.0),
                6: (0.12, 0.16, 0.23, 1.0),
                7: (0.15, 0.39, 0.92, 1.0),
                8: (0.86, 0.15, 0.15, 1.0),
                9: (0.09, 0.64, 0.29, 1.0),
                15: (0.92, 0.35, 0.05, 1.0),
                16: (0.79, 0.54, 0.02, 1.0),
                17: (0.08, 0.50, 0.24, 1.0),
                35: (0.49, 0.13, 0.81, 1.0),
                53: (0.42, 0.13, 0.66, 1.0),
            }
        )


def _render_2d_mol_png_array(
    mol: Chem.Mol,
    theme: ThemeName,
    width: int = 720,
    height: int = 440,
) -> NDArray[np.float64]:
    """Render a crisp 2D chemical structure to an RGBA numpy array via RDKit Cairo."""
    mol_2d = _prepare_2d_mol(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    _configure_rdkit_draw_options(drawer.drawOptions(), theme)
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return np.asarray(mpimg.imread(io.BytesIO(png_bytes)), dtype=np.float64)


def _render_2d_mol_svg(
    mol: Chem.Mol,
    theme: ThemeName = "dark",
    width: int = 320,
    height: int = 190,
) -> str:
    """Render a crisp transparent-ready 2D chemical structure SVG string for HTML HUDs."""
    mol_2d = _prepare_2d_mol(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts: Any = drawer.drawOptions()
    _configure_rdkit_draw_options(opts, theme)
    opts.clearBackground = False
    drawer.DrawMolecule(mol_2d)
    drawer.FinishDrawing()
    return str(drawer.GetDrawingText())


# ---------------------------------------------------------------------------
# Metric Badge Formatting Helpers
# ---------------------------------------------------------------------------


def _build_metric_cards(
    mol: Chem.Mol,
    metrics: Mapping[str, Any] | None,
    theme: ThemeName,
) -> list[tuple[str, str, str, str]]:
    """Format up to 8 KPI cards ``(label, value_str, subtitle, accent_hex)`` for the side panel."""
    m = dict(metrics) if metrics else {}
    cards: list[tuple[str, str, str, str]] = []

    green = "#34D399" if theme == "dark" else "#059669"
    cyan = "#38BDF8" if theme == "dark" else "#0284C7"
    amber = "#FBBF24" if theme == "dark" else "#D97706"
    rose = "#FB7185" if theme == "dark" else "#E11D48"
    violet = "#A78BFA" if theme == "dark" else "#7C3AED"
    slate = "#94A3B8" if theme == "dark" else "#475569"

    # 1) AutoDock Vina ΔG
    vina = m.get("vina_kcal", m.get("vina_affinity"))
    native_vina = m.get("native_vina_kcal", m.get("native_vina"))
    if vina is not None and math.isfinite(float(vina)):
        v = float(vina)
        sub = "kcal/mol"
        if native_vina is not None and math.isfinite(float(native_vina)):
            delta = v - float(native_vina)
            sub = f"vs {float(native_vina):.2f} (Δ{delta:+.2f})"
        accent = green if v <= -8.0 else (cyan if v <= -6.5 else amber)
        cards.append(("Vina ΔG", f"{v:.2f}", sub, accent))

    # 2) Pfizer CNS MPO (0–6)
    cns_mpo = m.get("cns_mpo")
    if cns_mpo is not None and math.isfinite(float(cns_mpo)):
        val = float(cns_mpo)
        accent = green if val >= 4.5 else (amber if val >= 3.0 else rose)
        cards.append(("CNS MPO", f"{val:.2f} / 6", "Pfizer CNS score", accent))

    # 3) P(BBB+)
    bbb = m.get("bbb_prob", m.get("p_bbb"))
    if bbb is not None and math.isfinite(float(bbb)):
        val = float(bbb)
        accent = green if val >= 0.7 else (amber if val >= 0.4 else rose)
        cards.append(("P(BBB+)", f"{val * 100:.0f}%", "Brain penetration", accent))

    # 4) P(Pgp-)
    pgp = m.get("pgp_minus_prob", m.get("p_pgp_minus"))
    if pgp is not None and math.isfinite(float(pgp)):
        val = float(pgp)
        accent = green if val >= 0.65 else (amber if val >= 0.4 else rose)
        cards.append(("P(Pgp−)", f"{val * 100:.0f}%", "Low efflux risk", accent))

    # 5) P(hERG)
    herg = m.get("herg_prob", m.get("p_herg"))
    if herg is not None and math.isfinite(float(herg)):
        val = float(herg)
        accent = green if val <= 0.30 else (amber if val <= 0.55 else rose)
        cards.append(("P(hERG)", f"{val * 100:.0f}%", "Cardiotox risk", accent))

    # 6) QED (auto-computed via RDKit if absent)
    qed_val = m.get("qed")
    if qed_val is None:
        try:
            qed_val = float(QED.qed(Chem.RemoveHs(mol)))
        except Exception:
            qed_val = None
    if qed_val is not None and math.isfinite(float(qed_val)):
        val = float(qed_val)
        accent = green if val >= 0.65 else (cyan if val >= 0.45 else amber)
        cards.append(("QED", f"{val:.3f}", "Drug-likeness", accent))

    # 7) SA Score
    sa_val = m.get("sa", m.get("sa_score"))
    if sa_val is not None and math.isfinite(float(sa_val)):
        val = float(sa_val)
        accent = green if val <= 3.2 else (amber if val <= 4.5 else rose)
        cards.append(("SA Score", f"{val:.2f}", "Synth. access.", accent))

    # 8) PoseBusters 3D Physical Plausibility
    pb = m.get("posebusters_pass", m.get("pb_pass"))
    if pb is not None:
        ok = bool(pb)
        cards.append(
            ("PoseBusters", "PASS" if ok else "FAIL", "3D physics audit", green if ok else rose)
        )

    # 9) PLIF Recovery / Self-Dock RMSD
    plif = m.get("plif_tanimoto")
    if plif is not None and math.isfinite(float(plif)):
        cards.append(("PLIF Recov.", f"{float(plif):.2f}", "3D contact Tanimoto", violet))

    rmsd = m.get("rmsd_angstrom", m.get("rmsd"))
    if rmsd is not None and math.isfinite(float(rmsd)):
        val = float(rmsd)
        cards.append(
            ("Pose RMSD", f"{val:.2f} Å", "Heavy-atom sym.", green if val <= 2.0 else amber)
        )

    # Fill out with RDKit physicochemical properties if fewer than 4 cards were supplied
    if len(cards) < 4:
        mol_noh = Chem.RemoveHs(mol)
        mw = float(_DESCRIPTORS.MolWt(mol_noh))
        logp = float(_DESCRIPTORS.MolLogP(mol_noh))
        tpsa = float(_DESCRIPTORS.TPSA(mol_noh))
        hbd = int(_DESCRIPTORS.NumHDonors(mol_noh))
        hba = int(_DESCRIPTORS.NumHAcceptors(mol_noh))
        for item in [
            ("Mol. Weight", f"{mw:.1f}", "Da (Ro5 ≤ 500)", cyan if mw <= 500 else amber),
            ("cLogP", f"{logp:.2f}", "Lipophilicity", cyan if -0.5 <= logp <= 5.0 else amber),
            ("TPSA", f"{tpsa:.1f} Å²", f"HBD {hbd} / HBA {hba}", violet if tpsa <= 90 else slate),
        ]:
            if len(cards) < 6:
                cards.append(item)

    return cards[:8]


# ---------------------------------------------------------------------------
# Mode B — 300-DPI Publication Static 3D Renderers
# ---------------------------------------------------------------------------


def _draw_side_panel(
    ax_2d: Axes,
    ax_info: Axes,
    mol: Chem.Mol,
    *,
    theme: ThemeName,
    metrics: Mapping[str, Any] | None,
    interactions: InteractionProfile | None,
    smiles: str,
) -> None:
    """Render the right-hand studio HUD: 2D RDKit structure, KPI cards, and contact list."""
    card_bg = "#0F172A" if theme == "dark" else "#F8FAFC"
    card_edge = "#1E293B" if theme == "dark" else "#CBD5E1"
    title_col = "#F8FAFC" if theme == "dark" else "#0F172A"
    sub_col = "#94A3B8" if theme == "dark" else "#475569"

    # --- Top Card: Crisp 2D Chemical Structure ---
    ax_2d.set_facecolor(card_bg)
    img_2d = _render_2d_mol_png_array(mol, theme=theme, width=760, height=430)
    ax_2d.imshow(img_2d)
    ax_2d.set_xticks([])
    ax_2d.set_yticks([])
    for spine in ax_2d.spines.values():
        spine.set_edgecolor(card_edge)
        spine.set_linewidth(1.1)
    ax_2d.set_title(
        "2D CHEMICAL GRAPH",
        loc="left",
        fontsize=8.2,
        fontweight="bold",
        color=sub_col,
        pad=5.0,
    )
    if smiles:
        disp_smi = smiles if len(smiles) <= 44 else smiles[:41] + "..."
        ax_2d.set_xlabel(disp_smi, fontsize=6.6, family="monospace", color=sub_col, labelpad=4.0)

    # --- Bottom Panel: Metric KPI Badges + Key 3D Interactions ---
    ax_info.set_facecolor("#0B1120" if theme == "dark" else "#FFFFFF")
    ax_info.set_xlim(0.0, 1.0)
    ax_info.set_ylim(0.0, 1.0)
    ax_info.axis("off")

    cards = _build_metric_cards(mol, metrics, theme)
    n_rows = max(1, (len(cards) + 1) // 2)
    has_table = interactions is not None
    top_y = 0.99
    if has_table:
        card_h = 0.120 if n_rows <= 3 else 0.106
        gap_y = 0.017
    else:
        card_h = 0.185 if n_rows <= 3 else 0.160
        gap_y = 0.024
    card_w = 0.48
    gap_x = 0.04

    ax_info.text(
        0.0,
        top_y,
        "MEDICINAL CHEMISTRY & 3D METRICS",
        fontsize=8.2,
        fontweight="bold",
        color=sub_col,
        va="top",
    )
    grid_top = top_y - 0.044

    for idx, (label, val_str, sub_str, accent) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = col * (card_w + gap_x)
        y0 = grid_top - (row + 1) * card_h - row * gap_y

        ax_info.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.006,rounding_size=0.020",
                facecolor=card_bg,
                edgecolor=card_edge,
                linewidth=0.95,
            )
        )
        # Colored vertical accent indicator bar
        ax_info.add_patch(
            FancyBboxPatch(
                (x0 + 0.011, y0 + card_h * 0.14),
                0.012,
                card_h * 0.72,
                boxstyle="round,pad=0.0,rounding_size=0.006",
                facecolor=accent,
                edgecolor="none",
            )
        )
        ax_info.text(
            x0 + 0.044,
            y0 + card_h * 0.86,
            label.upper(),
            fontsize=6.3,
            fontweight="bold",
            color=sub_col,
            va="top",
        )
        ax_info.text(
            x0 + 0.044,
            y0 + card_h * 0.48,
            val_str,
            fontsize=11.0 if not has_table else (10.2 if n_rows <= 3 else 9.4),
            fontweight="bold",
            color=title_col,
            va="center",
        )
        ax_info.text(
            x0 + 0.044,
            y0 + card_h * 0.11,
            sub_str,
            fontsize=5.9,
            color=sub_col,
            va="bottom",
        )

    # --- Key 3D Interactions Table beneath KPI cards ---
    table_top = grid_top - n_rows * card_h - (n_rows - 1) * gap_y - 0.036
    if interactions is not None and table_top > 0.14:
        ax_info.text(
            0.0,
            table_top,
            f"3D NON-COVALENT CONTACTS ({len(interactions.interactions)})",
            fontsize=8.2,
            fontweight="bold",
            color=sub_col,
            va="top",
        )
        y_cursor = table_top - 0.042
        row_h = 0.046
        max_rows = max(1, int((y_cursor - 0.008) / row_h))
        shown = interactions.interactions[:max_rows]

        if not shown:
            ax_info.text(
                0.02,
                y_cursor - 0.02,
                "No geometric contacts within cutoffs.",
                fontsize=7.2,
                color=sub_col,
                va="top",
            )
        else:
            for rec in shown:
                meta = _INTERACTION_META.get(rec.kind, _INTERACTION_META["hbond"])
                col_hex = meta[theme]
                ax_info.add_patch(
                    FancyBboxPatch(
                        (0.0, y_cursor - row_h + 0.008),
                        1.0,
                        row_h - 0.008,
                        boxstyle="round,pad=0.004,rounding_size=0.015",
                        facecolor=card_bg,
                        edgecolor=card_edge,
                        linewidth=0.65,
                    )
                )
                ax_info.plot(
                    [0.032],
                    [y_cursor - row_h * 0.44],
                    marker="o",
                    markersize=5.2,
                    color=col_hex,
                )
                ax_info.text(
                    0.068,
                    y_cursor - row_h * 0.44,
                    meta["short"],
                    fontsize=6.8,
                    fontweight="bold",
                    color=col_hex,
                    va="center",
                )
                ax_info.text(
                    0.22,
                    y_cursor - row_h * 0.44,
                    f"{rec.protein_residue} ({rec.protein_atom})",
                    fontsize=7.0,
                    fontweight="normal",
                    color=title_col,
                    va="center",
                    family="monospace",
                )
                dist_txt = f"{rec.distance_angstrom:.2f} Å"
                if rec.angle_deg is not None:
                    dist_txt += f" · {rec.angle_deg:.0f}°"
                ax_info.text(
                    0.97,
                    y_cursor - row_h * 0.44,
                    dist_txt,
                    fontsize=6.8,
                    color=sub_col,
                    va="center",
                    ha="right",
                    family="monospace",
                )
                y_cursor -= row_h


def render_static_3d_molecule_png(
    mol_3d: Chem.Mol,
    out_path: str | Path,
    *,
    title: str = "3D Molecular Conformer",
    subtitle: str = "",
    theme: ThemeName = "dark",
    metrics: Mapping[str, Any] | None = None,
    show_hydrogens: bool = False,
    show_side_panel: bool = True,
    ligand_carbon_color: str | None = None,
    elev_deg: float = 12.0,
    azim_deg: float = 10.0,
) -> Path:
    """Render a standalone 300-DPI publication 3D molecular visualization PNG.

    Uses a custom depth-sorted, pseudo-Phong specular-shaded 3D engine with automatic
    PCA camera framing and explicit single/double/triple/aromatic cylindrical bonds.
    """
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        raise ValueError("mol_3d must be an RDKit Mol containing a 3D conformer.")

    draw_mol = Chem.Mol(mol_3d) if show_hydrogens else Chem.RemoveHs(Chem.Mol(mol_3d))
    conf = draw_mol.GetConformer()
    n_atoms = draw_mol.GetNumAtoms()
    if n_atoms == 0:
        raise ValueError("mol_3d contains no atoms to render.")

    coords = np.asarray(conf.GetPositions(), dtype=np.float64)
    cam = _compute_optimal_camera(coords, elev_deg=elev_deg, azim_deg=azim_deg)

    bg_hex = "#0B1120" if theme == "dark" else "#FFFFFF"
    bg_rgb = _hex_to_rgb(bg_hex)
    title_col = "#F8FAFC" if theme == "dark" else "#0F172A"
    sub_col = "#94A3B8" if theme == "dark" else "#475569"

    if show_side_panel:
        fig = plt.figure(figsize=(12.4, 6.6), facecolor=bg_hex)
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[2.05, 1.0],
            height_ratios=[0.95, 1.25],
            wspace=0.06,
            hspace=0.14,
            left=0.02,
            right=0.98,
            top=0.96,
            bottom=0.04,
        )
        ax = fig.add_subplot(gs[:, 0])
        ax_2d = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(8.6, 6.8), facecolor=bg_hex)
        ax = fig.add_axes((0.02, 0.02, 0.96, 0.96))
        ax_2d = None
        ax_info = None

    ax.set_facecolor(bg_hex)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    proj_xy, z_vals = cam.project(coords)
    z_min = float(np.min(z_vals)) - 1.0
    z_max = float(np.max(z_vals)) + 1.0

    # Subtle smooth feathered radial backdrop glow under the molecule in dark studio mode
    if theme == "dark":
        cx, cy = float(np.mean(proj_xy[:, 0])), float(np.mean(proj_xy[:, 1]))
        span = max(
            float(np.ptp(proj_xy[:, 0])),
            float(np.ptp(proj_xy[:, 1])),
            4.5,
        )
        for step_g in range(10):
            frac = 1.0 - step_g / 10.0
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=span * (0.20 + 0.65 * frac),
                    facecolor="#1D4ED8",
                    edgecolor="none",
                    alpha=0.015,
                    zorder=0.1,
                )
            )

    # Assemble depth-sorted draw queue: (z_depth, primitive_kind, payload)
    queue: list[tuple[float, str, dict[str, Any]]] = []
    for bond in draw_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        el1 = draw_mol.GetAtomWithIdx(i).GetSymbol()
        el2 = draw_mol.GetAtomWithIdx(j).GetSymbol()
        z_mid = 0.5 * (float(z_vals[i]) + float(z_vals[j])) - 0.06
        _, _, _, s1 = cam.project_point(coords[i])
        _, _, _, s2 = cam.project_point(coords[j])
        queue.append(
            (
                z_mid,
                "bond",
                {
                    "xy1": (float(proj_xy[i, 0]), float(proj_xy[i, 1])),
                    "xy2": (float(proj_xy[j, 0]), float(proj_xy[j, 1])),
                    "rgb1": _apply_depth_fog(
                        _element_color(el1, theme, carbon_override=ligand_carbon_color),
                        bg_rgb,
                        float(z_vals[i]),
                        z_min,
                        z_max,
                        strength=0.22,
                    ),
                    "rgb2": _apply_depth_fog(
                        _element_color(el2, theme, carbon_override=ligand_carbon_color),
                        bg_rgb,
                        float(z_vals[j]),
                        z_min,
                        z_max,
                        strength=0.22,
                    ),
                    "bond_type": bond.GetBondType(),
                    "is_aromatic": bond.GetIsAromatic(),
                    "scale": 0.5 * (s1 + s2),
                },
            )
        )

    for idx in range(n_atoms):
        atom = draw_mol.GetAtomWithIdx(idx)
        el = atom.GetSymbol().upper()
        x2, y2, zc, sc = cam.project_point(coords[idx])
        rgb = _apply_depth_fog(
            _element_color(el, theme, carbon_override=ligand_carbon_color),
            bg_rgb,
            zc,
            z_min,
            z_max,
            strength=0.22,
        )
        rad = _ATOM_VISUAL_RADIUS.get(el, 0.36) * sc
        queue.append((zc, "atom", {"x": x2, "y": y2, "r": rad, "rgb": rgb}))

    queue.sort(key=lambda item: item[0])
    for step, (_, prim_kind, payload) in enumerate(queue):
        zo = 2.0 + step * 0.01
        if prim_kind == "bond":
            _draw_multi_order_bond(
                ax,
                payload["xy1"],
                payload["xy2"],
                payload["rgb1"],
                payload["rgb2"],
                bg_rgb,
                bond_type=payload["bond_type"],
                is_aromatic=payload["is_aromatic"],
                persp_scale=payload["scale"],
                zorder=zo,
                base_width_pt=5.4,
            )
        else:
            _draw_phong_sphere(
                ax,
                payload["x"],
                payload["y"],
                payload["r"],
                payload["rgb"],
                bg_rgb,
                zorder=zo,
            )

    # Frame cleanly with generous studio margins
    pad = max(1.8, 0.24 * max(float(np.ptp(proj_xy[:, 0])), float(np.ptp(proj_xy[:, 1])), 4.0))
    xmin, xmax = float(np.min(proj_xy[:, 0])) - pad, float(np.max(proj_xy[:, 0])) + pad
    ymin, ymax = float(np.min(proj_xy[:, 1])) - pad, float(np.max(proj_xy[:, 1])) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Studio header overlay
    ax.text(
        0.025,
        0.965,
        "GENEUCLIDEAN v3.1  •  3D MOLECULAR STUDIO",
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="bold",
        color="#38BDF8" if theme == "dark" else "#0284C7",
        va="top",
    )
    ax.text(
        0.025,
        0.922,
        title,
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        color=title_col,
        va="top",
    )
    if subtitle:
        ax.text(
            0.025,
            0.872,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.6,
            color=sub_col,
            va="top",
        )

    # Subtle 2 Å scale bar bottom-right
    sb_x1 = xmax - 0.55
    sb_x0 = sb_x1 - 2.0
    sb_y = ymin + 0.45
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color=sub_col, lw=2.0, zorder=100)
    ax.text(
        0.5 * (sb_x0 + sb_x1),
        sb_y + 0.18,
        "2.0 Å",
        fontsize=7.5,
        color=sub_col,
        ha="center",
        va="bottom",
        family="monospace",
        zorder=100,
    )

    if show_side_panel and ax_2d is not None and ax_info is not None:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol_3d), isomericSmiles=True)
        _draw_side_panel(
            ax_2d,
            ax_info,
            mol_3d,
            theme=theme,
            metrics=metrics,
            interactions=None,
            smiles=smi,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_DPI, facecolor=bg_hex, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def render_static_3d_complex_png(
    pocket_pdb_path: str | Path,
    ligand_mol: Chem.Mol,
    out_path: str | Path,
    *,
    interactions: InteractionProfile | None = None,
    native_ligand_mol: Chem.Mol | None = None,
    title: str = "3D Protein-Ligand Binding Complex",
    pdb_id: str = "",
    disease: str = "",
    theme: ThemeName = "dark",
    metrics: Mapping[str, Any] | None = None,
    residue_attention: Mapping[str, float] | None = None,
    ligand_carbon_color: str | None = None,
    max_hydrophobic_lines: int = 4,
    elev_deg: float = 11.0,
    azim_deg: float = 9.0,
) -> Path:
    """Render a 300-DPI publication-grade 3D protein-ligand complex PNG with HUD side panel.

    Features:
    - Automatic occlusion-free PCA camera oriented so the protein pocket opens toward the viewer.
    - Smooth Catmull-Rom Cα backbone trace + translucent cavity glow (colored by 3D Perceiver
      attention weights when ``residue_attention`` is supplied).
    - Depth-sorted pseudo-Phong CPK ligand spheres & multi-order cylindrical bonds.
    - Optional translucent ghost overlay of the native co-crystal ligand.
    - Key interacting side-chain sticks, dashed 3D non-covalent interaction vectors with
      Ångström distance badges, and collision-free residue callouts.
    - Side panel with 2D RDKit structure, medicinal chemistry KPI badges, and contact table.
    """
    if ligand_mol is None or ligand_mol.GetNumConformers() == 0:
        raise ValueError("ligand_mol must be an RDKit Mol containing a 3D conformer.")

    pocket_path = Path(pocket_pdb_path)
    if not pocket_path.is_file():
        raise FileNotFoundError(f"Protein pocket PDB not found: {pocket_path}")

    profile = (
        interactions
        if interactions is not None
        else profile_3d_interactions(pocket_path, ligand_mol)
    )
    pocket_geom = _load_pocket_geometry(pocket_pdb_path=pocket_path)

    lig_noh = Chem.RemoveHs(Chem.Mol(ligand_mol))
    lig_coords = np.asarray(lig_noh.GetConformer().GetPositions(), dtype=np.float64)
    pocket_all_xyz = (
        np.stack([a.xyz for a in pocket_geom.atoms], axis=0) if pocket_geom.atoms else None
    )
    contact_midpoints = (
        np.stack(
            [
                0.5 * (np.asarray(r.protein_xyz) + np.asarray(r.ligand_xyz))
                for r in profile.interactions
            ],
            axis=0,
        )
        if profile.interactions
        else None
    )

    cam = _compute_optimal_camera(
        lig_coords,
        pocket_coords=pocket_all_xyz,
        contact_coords=contact_midpoints,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
    )

    bg_hex = "#0B1120" if theme == "dark" else "#FFFFFF"
    bg_rgb = _hex_to_rgb(bg_hex)
    title_col = "#F8FAFC" if theme == "dark" else "#0F172A"
    sub_col = "#94A3B8" if theme == "dark" else "#475569"
    pill_bg = "#0F172A" if theme == "dark" else "#F8FAFC"

    fig = plt.figure(figsize=(13.6, 7.6), facecolor=bg_hex)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[2.12, 1.0],
        height_ratios=[0.78, 1.52],
        wspace=0.055,
        hspace=0.14,
        left=0.018,
        right=0.985,
        top=0.965,
        bottom=0.035,
    )
    ax = fig.add_subplot(gs[:, 0])
    ax_2d = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    ax.set_facecolor(bg_hex)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Select which interactions and residues to highlight cleanly in 3D
    # Deduplicate overlapping 3D vectors on the exact same atom pair (e.g. salt bridge + H-bond)
    display_interactions: list[InteractionRecord] = []
    seen_atom_pairs: set[tuple[str, str, int]] = set()
    hydro_count = 0
    for rec in profile.interactions:
        if rec.kind == "hydrophobic":
            if hydro_count >= max_hydrophobic_lines:
                continue
            hydro_count += 1
        else:
            pair_key = (rec.protein_residue, rec.protein_atom, rec.ligand_atom_idx)
            if pair_key in seen_atom_pairs:
                continue
            seen_atom_pairs.add(pair_key)
        display_interactions.append(rec)

    active_residues: list[str] = []
    for rec in display_interactions:
        if rec.protein_residue not in active_residues:
            active_residues.append(rec.protein_residue)
    active_residues = active_residues[:10]

    # Identify global depth range across ligand + active sidechains for smooth atmospheric fog
    depth_samples = [cam.transform(lig_coords)[:, 2]]
    for res_id in active_residues:
        r_atoms = pocket_geom.by_residue.get(res_id, [])
        if r_atoms:
            depth_samples.append(cam.transform(np.stack([a.xyz for a in r_atoms], axis=0))[:, 2])
    all_z = np.concatenate(depth_samples)
    z_min = float(np.min(all_z)) - 3.0
    z_max = float(np.max(all_z)) + 1.2

    # --- Layer 1: Soft Feathered Pocket Cavity Glow / 3D Perceiver Attention Halo ---
    if pocket_geom.atoms:
        for p_atom in pocket_geom.atoms:
            d_to_lig = float(np.min(np.linalg.norm(lig_coords - p_atom.xyz, axis=1)))
            if d_to_lig > 6.6:
                continue
            x2, y2, zc, sc = cam.project_point(p_atom.xyz)
            attn = _lookup_residue_attention(p_atom.residue_id, p_atom.resseq, residue_attention)
            if attn is not None:
                halo_rgb = _attention_color_rgb(attn, theme=theme)
                base_alpha = 0.018 + 0.036 * max(0.0, min(1.0, attn))
            else:
                halo_rgb = _hex_to_rgb("#312E81") if theme == "dark" else _hex_to_rgb("#CBD5E1")
                base_alpha = 0.018 if theme == "dark" else 0.024
            # Three concentric feathered rings eliminate hard circular edges
            for r_mult, a_mult in ((1.42, 0.42), (0.98, 0.72), (0.56, 1.00)):
                ax.add_patch(
                    Circle(
                        (x2, y2),
                        radius=r_mult * sc,
                        facecolor=tuple(halo_rgb),
                        edgecolor="none",
                        alpha=base_alpha * a_mult,
                        zorder=0.4,
                    )
                )

    # --- Build Painter's-Algorithm Depth-Sorted Primitive Queue ---
    queue: list[tuple[float, str, dict[str, Any]]] = []

    # 1) Smooth Cα Protein Backbone Ribbon Traces near the binding site
    ribbon_default = _hex_to_rgb("#475569") if theme == "dark" else _hex_to_rgb("#94A3B8")
    for trace in pocket_geom.ca_traces:
        ca_pts = np.stack([item[2] for item in trace], axis=0)
        # Keep only traces that approach within 10.5 Å of the ligand
        min_d = float(np.min(np.linalg.norm(ca_pts[:, None, :] - lig_coords[None, :, :], axis=-1)))
        if min_d > 10.5:
            continue
        spline_pts = _catmull_rom_spline(ca_pts, subdivisions=6)
        n_ctrl = len(trace)
        for seg_idx in range(len(spline_pts) - 1):
            p1 = spline_pts[seg_idx]
            p2 = spline_pts[seg_idx + 1]
            mid_pt = 0.5 * (p1 + p2)
            # Clip far backbone segments cleanly so viewport stays focused on the pocket
            if float(np.min(np.linalg.norm(lig_coords - mid_pt, axis=1))) > 11.0:
                continue
            ctrl_idx = min(n_ctrl - 1, int(round(seg_idx / 6.0)))
            res_id, resseq, _ = trace[ctrl_idx]
            attn = _lookup_residue_attention(res_id, resseq, residue_attention)
            base_col = (
                _attention_color_rgb(attn, theme=theme) if attn is not None else ribbon_default
            )
            x1, y1, z1, sc1 = cam.project_point(p1)
            x2, y2, z2, sc2 = cam.project_point(p2)
            zm = 0.5 * (z1 + z2)
            fog_col = _apply_depth_fog(base_col, bg_rgb, zm, z_min, z_max, strength=0.48)
            queue.append(
                (
                    zm - 0.5,
                    "ribbon",
                    {
                        "p1": (x1, y1),
                        "p2": (x2, y2),
                        "rgb": fog_col,
                        "w": 4.2 * 0.5 * (sc1 + sc2),
                        "alpha": 0.42 if attn is None else 0.65,
                    },
                )
            )

    # 2) Key Interacting Pocket Side-Chain Sticks & Aromatic Ring Shading
    res_carbon = _hex_to_rgb("#94A3B8") if theme == "dark" else _hex_to_rgb("#64748B")
    for res_id in active_residues:
        r_atoms = pocket_geom.by_residue.get(res_id, [])
        if not r_atoms:
            continue
        res_coords = np.stack([a.xyz for a in r_atoms], axis=0)
        res_xy, res_z = cam.project(res_coords)

        # Translucent fill for interacting aromatic rings (PHE/TYR/TRP/HIS)
        for poly3d in pocket_geom.aromatic_polygons.get(res_id, []):
            poly_xy, poly_z = cam.project(poly3d)
            zm = float(np.mean(poly_z))
            ring_col = _hex_to_rgb("#FBBF24") if theme == "dark" else _hex_to_rgb("#D97706")
            queue.append(
                (
                    zm - 0.15,
                    "ring_poly",
                    {"xy": poly_xy, "rgb": ring_col, "alpha": 0.16},
                )
            )

        # Intra-residue covalent bonds (heavy-atom distance <= 1.92 Å)
        n_ra = len(r_atoms)
        for i in range(n_ra):
            for j in range(i + 1, n_ra):
                dist = float(np.linalg.norm(r_atoms[i].xyz - r_atoms[j].xyz))
                if dist <= 1.92:
                    zm = 0.5 * (float(res_z[i]) + float(res_z[j])) - 0.08
                    c1 = (
                        res_carbon
                        if r_atoms[i].element == "C"
                        else _blend_rgb(_element_color(r_atoms[i].element, theme), res_carbon, 0.25)
                    )
                    c2 = (
                        res_carbon
                        if r_atoms[j].element == "C"
                        else _blend_rgb(_element_color(r_atoms[j].element, theme), res_carbon, 0.25)
                    )
                    c1 = _apply_depth_fog(c1, bg_rgb, float(res_z[i]), z_min, z_max, strength=0.36)
                    c2 = _apply_depth_fog(c2, bg_rgb, float(res_z[j]), z_min, z_max, strength=0.36)
                    queue.append(
                        (
                            zm,
                            "pocket_bond",
                            {
                                "xy1": (float(res_xy[i, 0]), float(res_xy[i, 1])),
                                "xy2": (float(res_xy[j, 0]), float(res_xy[j, 1])),
                                "rgb1": c1,
                                "rgb2": c2,
                            },
                        )
                    )

        # Subtle shaded end-caps on polar heteroatoms of active sidechains
        for pa in r_atoms:
            if pa.element in {"N", "O", "S"}:
                x2, y2, zc, sc = cam.project_point(pa.xyz)
                rgb = _apply_depth_fog(
                    _blend_rgb(_element_color(pa.element, theme), res_carbon, 0.18),
                    bg_rgb,
                    zc,
                    z_min,
                    z_max,
                    strength=0.35,
                )
                queue.append(
                    (
                        zc - 0.04,
                        "pocket_atom",
                        {"x": x2, "y": y2, "r": 0.20 * sc, "rgb": rgb},
                    )
                )

    # 3) Optional Native Co-Crystal Ghost Ligand Overlay
    if native_ligand_mol is not None and native_ligand_mol.GetNumConformers() > 0:
        nat_noh = Chem.RemoveHs(Chem.Mol(native_ligand_mol))
        nat_coords = np.asarray(nat_noh.GetConformer().GetPositions(), dtype=np.float64)
        nat_xy, nat_z = cam.project(nat_coords)
        ghost_rgb = _hex_to_rgb("#94A3B8") if theme == "dark" else _hex_to_rgb("#64748B")
        for bond in nat_noh.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            zm = 0.5 * (float(nat_z[i]) + float(nat_z[j])) - 0.12
            queue.append(
                (
                    zm,
                    "ghost_bond",
                    {
                        "xy1": (float(nat_xy[i, 0]), float(nat_xy[i, 1])),
                        "xy2": (float(nat_xy[j, 0]), float(nat_xy[j, 1])),
                        "rgb": ghost_rgb,
                    },
                )
            )

    # 4) Main Ligand 3D Bonds & Pseudo-Phong Shaded CPK Spheres
    lig_xy, lig_z = cam.project(lig_coords)
    for bond in lig_noh.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        el1 = lig_noh.GetAtomWithIdx(i).GetSymbol()
        el2 = lig_noh.GetAtomWithIdx(j).GetSymbol()
        zm = 0.5 * (float(lig_z[i]) + float(lig_z[j])) - 0.05
        _, _, _, s1 = cam.project_point(lig_coords[i])
        _, _, _, s2 = cam.project_point(lig_coords[j])
        queue.append(
            (
                zm,
                "lig_bond",
                {
                    "xy1": (float(lig_xy[i, 0]), float(lig_xy[i, 1])),
                    "xy2": (float(lig_xy[j, 0]), float(lig_xy[j, 1])),
                    "rgb1": _apply_depth_fog(
                        _element_color(el1, theme, carbon_override=ligand_carbon_color),
                        bg_rgb,
                        float(lig_z[i]),
                        z_min,
                        z_max,
                        strength=0.18,
                    ),
                    "rgb2": _apply_depth_fog(
                        _element_color(el2, theme, carbon_override=ligand_carbon_color),
                        bg_rgb,
                        float(lig_z[j]),
                        z_min,
                        z_max,
                        strength=0.18,
                    ),
                    "bond_type": bond.GetBondType(),
                    "is_aromatic": bond.GetIsAromatic(),
                    "scale": 0.5 * (s1 + s2),
                },
            )
        )

    for idx in range(lig_noh.GetNumAtoms()):
        el = lig_noh.GetAtomWithIdx(idx).GetSymbol().upper()
        x2, y2, zc, sc = cam.project_point(lig_coords[idx])
        rgb = _apply_depth_fog(
            _element_color(el, theme, carbon_override=ligand_carbon_color),
            bg_rgb,
            zc,
            z_min,
            z_max,
            strength=0.18,
        )
        rad = _ATOM_VISUAL_RADIUS.get(el, 0.36) * sc
        queue.append((zc, "lig_atom", {"x": x2, "y": y2, "r": rad, "rgb": rgb}))

    # 5) 3D Non-Covalent Interaction Dashed Vectors
    for rec in display_interactions:
        p3 = np.asarray(rec.protein_xyz, dtype=np.float64)
        l3 = np.asarray(rec.ligand_xyz, dtype=np.float64)
        px, py, pz, _ = cam.project_point(p3)
        lx, ly, lz, _ = cam.project_point(l3)
        meta = _INTERACTION_META.get(rec.kind, _INTERACTION_META["hbond"])
        queue.append(
            (
                max(pz, lz) + 0.15,
                "interaction",
                {
                    "p": (px, py),
                    "l": (lx, ly),
                    "color": meta[theme],
                    "kind": rec.kind,
                    "dist": rec.distance_angstrom,
                },
            )
        )

    # Execute depth-sorted painter's pass
    dist_badge_anchors: list[tuple[float, float]] = []
    dist_badge_meta: list[tuple[str, str]] = []

    queue.sort(key=lambda item: item[0])
    for step, (_, prim_kind, payload) in enumerate(queue):
        zo = 2.0 + step * 0.01
        if prim_kind == "ribbon":
            p1, p2 = payload["p1"], payload["p2"]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=tuple(payload["rgb"]),
                lw=payload["w"],
                solid_capstyle="butt",
                alpha=payload["alpha"],
                zorder=zo,
            )
        elif prim_kind == "ring_poly":
            ax.add_patch(
                Polygon(
                    payload["xy"],
                    closed=True,
                    facecolor=tuple(payload["rgb"]),
                    edgecolor=tuple(payload["rgb"]),
                    linewidth=0.7,
                    alpha=payload["alpha"],
                    zorder=zo,
                )
            )
        elif prim_kind == "pocket_bond":
            xy1, xy2 = payload["xy1"], payload["xy2"]
            mid = (0.5 * (xy1[0] + xy2[0]), 0.5 * (xy1[1] + xy2[1]))
            ax.plot(
                [xy1[0], mid[0]],
                [xy1[1], mid[1]],
                color=tuple(payload["rgb1"]),
                lw=2.1,
                solid_capstyle="round",
                alpha=0.82,
                zorder=zo,
            )
            ax.plot(
                [mid[0], xy2[0]],
                [mid[1], xy2[1]],
                color=tuple(payload["rgb2"]),
                lw=2.1,
                solid_capstyle="round",
                alpha=0.82,
                zorder=zo,
            )
        elif prim_kind == "pocket_atom":
            _draw_phong_sphere(
                ax,
                payload["x"],
                payload["y"],
                payload["r"],
                payload["rgb"],
                bg_rgb,
                zorder=zo,
                alpha=0.88,
                specular=0.65,
            )
        elif prim_kind == "ghost_bond":
            xy1, xy2 = payload["xy1"], payload["xy2"]
            ax.plot(
                [xy1[0], xy2[0]],
                [xy1[1], xy2[1]],
                color=tuple(payload["rgb"]),
                lw=2.8,
                linestyle="--",
                solid_capstyle="round",
                alpha=0.38,
                zorder=zo,
            )
        elif prim_kind == "lig_bond":
            _draw_multi_order_bond(
                ax,
                payload["xy1"],
                payload["xy2"],
                payload["rgb1"],
                payload["rgb2"],
                bg_rgb,
                bond_type=payload["bond_type"],
                is_aromatic=payload["is_aromatic"],
                persp_scale=payload["scale"],
                zorder=zo,
                base_width_pt=4.5,
            )
        elif prim_kind == "lig_atom":
            _draw_phong_sphere(
                ax,
                payload["x"],
                payload["y"],
                payload["r"],
                payload["rgb"],
                bg_rgb,
                zorder=zo,
            )
        elif prim_kind == "interaction":
            px, py = payload["p"]
            lx, ly = payload["l"]
            col = payload["color"]
            is_hydro = payload["kind"] == "hydrophobic"
            if theme == "dark" and not is_hydro:
                ax.plot(
                    [px, lx],
                    [py, ly],
                    color=col,
                    lw=4.4,
                    alpha=0.22,
                    solid_capstyle="round",
                    zorder=zo,
                )
            ax.plot(
                [px, lx],
                [py, ly],
                color=col,
                lw=1.35 if is_hydro else 1.85,
                linestyle=(0, (3.2, 2.2)) if not is_hydro else (0, (2.0, 2.5)),
                alpha=0.72 if is_hydro else 0.96,
                zorder=zo + 0.001,
            )
            if not is_hydro:
                mx, my = 0.5 * (px + lx), 0.5 * (py + ly)
                dist_badge_anchors.append((mx, my))
                dist_badge_meta.append((f"{payload['dist']:.1f}Å", col))

    # --- Compute Balanced Viewport Framing around Ligand + Active Residues ---
    frame_pts = [lig_xy]
    for res_id in active_residues:
        r_atoms = pocket_geom.by_residue.get(res_id, [])
        if r_atoms:
            rxy, _ = cam.project(np.stack([a.xyz for a in r_atoms], axis=0))
            frame_pts.append(rxy)
    all_frame_xy = np.vstack(frame_pts)
    x_span = max(float(np.ptp(all_frame_xy[:, 0])), 7.0)
    y_span = max(float(np.ptp(all_frame_xy[:, 1])), 6.0)
    pad_x = max(1.8, 0.16 * x_span)
    pad_y_top = max(2.2, 0.22 * y_span)
    pad_y_bot = max(2.0, 0.19 * y_span)
    xmin = float(np.min(all_frame_xy[:, 0])) - pad_x
    xmax = float(np.max(all_frame_xy[:, 0])) + pad_x
    ymin = float(np.min(all_frame_xy[:, 1])) - pad_y_bot
    ymax = float(np.max(all_frame_xy[:, 1])) + pad_y_top
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Render de-overlapped midpoint Ångström distance badges
    if dist_badge_anchors:
        dist_placed = _deoverlap_label_positions(
            dist_badge_anchors,
            dist_badge_anchors,
            box_w=max(1.15, 0.075 * (xmax - xmin)),
            box_h=max(0.55, 0.044 * (ymax - ymin)),
            bounds=(xmin + 0.4, xmax - 0.4, ymin + 0.8, ymax - 1.3),
            iterations=25,
        )
        for (mx, my), (txt, col) in zip(dist_placed, dist_badge_meta, strict=True):
            ax.text(
                mx,
                my,
                txt,
                fontsize=6.2,
                fontweight="bold",
                family="monospace",
                color=col,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15,rounding_size=0.24",
                    "facecolor": bg_hex,
                    "edgecolor": col,
                    "linewidth": 0.65,
                    "alpha": 0.90,
                },
                zorder=90.0,
            )

    # --- Collision-Free Residue Callout Pills ---
    lig_center_2d = np.mean(lig_xy, axis=0)
    res_anchors: list[tuple[float, float]] = []
    res_init_pos: list[tuple[float, float]] = []
    res_labels: list[tuple[str, str]] = []

    for res_id in active_residues:
        # Pick the primary interaction anchor for this residue
        rec_match = next((r for r in display_interactions if r.protein_residue == res_id), None)
        if rec_match is None:
            continue
        ax_pt, ay_pt, _, _ = cam.project_point(np.asarray(rec_match.protein_xyz, dtype=np.float64))
        vec_out = np.array([ax_pt - lig_center_2d[0], ay_pt - lig_center_2d[1]], dtype=np.float64)
        v_norm = float(np.linalg.norm(vec_out))
        direction = vec_out / v_norm if v_norm > 1e-4 else np.array([0.0, 1.0])
        push_dist = max(1.35, 0.11 * max(x_span, y_span))
        res_anchors.append((ax_pt, ay_pt))
        res_init_pos.append(
            (ax_pt + float(direction[0]) * push_dist, ay_pt + float(direction[1]) * push_dist)
        )
        col_hex = _INTERACTION_META.get(rec_match.kind, _INTERACTION_META["hbond"])[theme]
        res_labels.append((_short_residue_label(res_id), col_hex))

    if res_labels:
        placed = _deoverlap_label_positions(
            res_anchors,
            res_init_pos,
            box_w=max(1.75, 0.125 * (xmax - xmin)),
            box_h=max(0.75, 0.062 * (ymax - ymin)),
            bounds=(xmin + 0.3, xmax - 0.3, ymin + 0.9, ymax - 1.4),
        )
        for (ax_pt, ay_pt), (lx_pt, ly_pt), (lbl_txt, col_hex) in zip(
            res_anchors, placed, res_labels, strict=True
        ):
            ax.plot(
                [ax_pt, lx_pt],
                [ay_pt, ly_pt],
                color=col_hex,
                lw=0.85,
                alpha=0.55,
                zorder=92.0,
            )
            ax.text(
                lx_pt,
                ly_pt,
                lbl_txt,
                fontsize=7.3,
                fontweight="bold",
                family="monospace",
                color=title_col,
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.24,rounding_size=0.30",
                    "facecolor": pill_bg,
                    "edgecolor": col_hex,
                    "linewidth": 1.15,
                    "alpha": 0.94,
                },
                zorder=95.0,
            )

    # --- Top Studio Title Banner ---
    ax.text(
        0.022,
        0.972,
        "GENEUCLIDEAN v3.1  •  3D PROTEIN-LIGAND COMPLEX STUDIO",
        transform=ax.transAxes,
        fontsize=7.6,
        fontweight="bold",
        color="#38BDF8" if theme == "dark" else "#0284C7",
        va="top",
        zorder=99.0,
    )
    ax.text(
        0.022,
        0.930,
        title,
        transform=ax.transAxes,
        fontsize=13.2,
        fontweight="bold",
        color=title_col,
        va="top",
        zorder=99.0,
    )
    meta_parts = [p for p in (f"PDB {pdb_id.upper()}" if pdb_id else "", disease) if p]
    if meta_parts:
        ax.text(
            0.022,
            0.884,
            "  •  ".join(meta_parts),
            transform=ax.transAxes,
            fontsize=8.6,
            color=sub_col,
            va="top",
            zorder=99.0,
        )

    # --- Bottom Interaction Channel Legend & Scale Bar ---
    active_kinds = [k for k in ALL_INTERACTION_KINDS if profile.counts_by_type.get(k, 0) > 0]
    if not active_kinds:
        active_kinds = ["hbond", "salt_bridge", "pi_stacking", "hydrophobic"]
    x_leg = 0.022
    for kind in active_kinds:
        meta = _INTERACTION_META[kind]
        col = meta[theme]
        cnt = profile.counts_by_type.get(kind, 0)
        label_str = f"{meta['label']} ({cnt})"
        ax.plot(
            [x_leg],
            [0.032],
            marker="o",
            markersize=5.4,
            color=col,
            transform=ax.transAxes,
            zorder=99.0,
        )
        ax.text(
            x_leg + 0.015,
            0.032,
            label_str,
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=title_col,
            va="center",
            zorder=99.0,
        )
        x_leg += 0.036 + 0.0096 * len(label_str)

    if native_ligand_mol is not None:
        ax.text(
            0.978,
            0.068,
            "--- Native Co-Crystal Ghost",
            transform=ax.transAxes,
            fontsize=7.0,
            color=sub_col,
            ha="right",
            va="center",
            zorder=99.0,
        )

    sb_x1 = xmax - 0.55
    sb_x0 = sb_x1 - 2.0
    sb_y = ymin + 0.42
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color=sub_col, lw=2.0, zorder=99.0)
    ax.text(
        0.5 * (sb_x0 + sb_x1),
        sb_y + 0.16,
        "2.0 Å",
        fontsize=7.3,
        color=sub_col,
        ha="center",
        va="bottom",
        family="monospace",
        zorder=99.0,
    )

    # Subtle attention color bar pill when 3D Perceiver attention weights are shown
    if residue_attention:
        ax.text(
            0.978,
            0.968,
            "3D PERCEIVER ATTENTION",
            transform=ax.transAxes,
            fontsize=6.6,
            fontweight="bold",
            color="#FBBF24" if theme == "dark" else "#D97706",
            ha="right",
            va="top",
            zorder=99.0,
        )
        for s_idx in range(18):
            t = s_idx / 17.0
            ax.add_patch(
                FancyBboxPatch(
                    (0.815 + t * 0.155, 0.928),
                    0.010,
                    0.014,
                    transform=ax.transAxes,
                    boxstyle="square,pad=0.0",
                    facecolor=tuple(_attention_color_rgb(t, theme=theme)),
                    edgecolor="none",
                    zorder=99.0,
                )
            )

    # --- Right-Hand Studio Side Panel ---
    smi = Chem.MolToSmiles(lig_noh, isomericSmiles=True)
    _draw_side_panel(
        ax_2d,
        ax_info,
        ligand_mol,
        theme=theme,
        metrics=metrics,
        interactions=profile,
        smiles=smi,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_DPI, facecolor=bg_hex, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Mode A — Interactive Browser 3Dmol.js WebGL Studio HTML5 Generator
# ---------------------------------------------------------------------------


def _serialize_showcase_spec(spec: ComplexShowcaseSpec) -> dict[str, Any]:
    """Resolve PDB, SDF, 2D SVG, interactions, and metrics into a clean JSON payload."""
    pdb_text = spec.pocket_pdb_text or ""
    if not pdb_text and spec.pocket_pdb_path is not None:
        pdb_path = Path(spec.pocket_pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Protein pocket PDB not found: {pdb_path}")
        pdb_text = pdb_path.read_text(encoding="utf-8")

    ligand_sdf = ""
    svg_2d = ""
    smiles = spec.smiles
    derived_metrics: dict[str, Any] = dict(spec.metrics)

    if spec.ligand_mol is not None:
        if spec.ligand_mol.GetNumConformers() == 0:
            raise ValueError(f"Spec '{spec.target_id}' ligand_mol has no 3D conformer.")
        ligand_sdf = str(Chem.MolToMolBlock(spec.ligand_mol))
        mol_noh = Chem.RemoveHs(Chem.Mol(spec.ligand_mol))
        if not smiles:
            smiles = Chem.MolToSmiles(mol_noh, isomericSmiles=True)
        svg_2d = _render_2d_mol_svg(spec.ligand_mol, theme="dark", width=330, height=185)
        if "qed" not in derived_metrics:
            with contextlib.suppress(Exception):
                derived_metrics["qed"] = round(float(QED.qed(mol_noh)), 3)
        if "mw" not in derived_metrics:
            derived_metrics["mw"] = round(float(_DESCRIPTORS.MolWt(mol_noh)), 1)
        if "logp" not in derived_metrics:
            derived_metrics["logp"] = round(float(_DESCRIPTORS.MolLogP(mol_noh)), 2)

    native_sdf = ""
    if spec.native_ligand_mol is not None and spec.native_ligand_mol.GetNumConformers() > 0:
        native_sdf = str(Chem.MolToMolBlock(spec.native_ligand_mol))

    profile = spec.interactions
    if profile is None and pdb_text and spec.ligand_mol is not None:
        profile = profile_3d_interactions_from_pdb_text(pdb_text, spec.ligand_mol)

    attn_clean = (
        {str(k): round(float(v), 4) for k, v in spec.residue_attention.items()}
        if spec.residue_attention
        else {}
    )

    return {
        "target_id": spec.target_id,
        "title": spec.title,
        "pdb_id": spec.pdb_id.upper(),
        "disease": spec.disease,
        "smiles": smiles,
        "notes": spec.notes,
        "pdb": pdb_text,
        "ligand_sdf": ligand_sdf,
        "native_sdf": native_sdf,
        "svg_2d": svg_2d,
        "metrics": derived_metrics,
        "attention": attn_clean,
        "profile": profile.to_dict() if profile is not None else InteractionProfile().to_dict(),
    }


def _safe_inline_json(obj: Any) -> str:
    """Serialize JSON safe for inline HTML ``<script type='application/json'>`` embedding."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/")


_INTERACTIVE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>__PAGE_TITLE__</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"
        onerror="var s=document.createElement('script');s.src='https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js';document.head.appendChild(s);"></script>
<style>
  :root {
    --bg: #050814;
    --panel: rgba(12, 19, 36, 0.86);
    --card: rgba(18, 28, 52, 0.88);
    --border: rgba(148, 163, 184, 0.16);
    --text: #F8FAFC;
    --muted: #94A3B8;
    --cyan: #38BDF8;
    --emerald: #34D399;
    --magenta: #F472B6;
    --gold: #FBBF24;
    --orange: #FB923C;
    --violet: #A78BFA;
    --rose: #FB7185;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: radial-gradient(circle at 35% 30%, #0F172A 0%, #050814 75%);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .studio-shell {
    display: grid;
    grid-template-columns: __GRID_COLS__;
    width: 100vw; height: 100vh;
  }
  /* Left Gallery Rail (shown in gallery mode) */
  .gallery-rail {
    display: __RAIL_DISPLAY__;
    flex-direction: column;
    background: rgba(8, 13, 27, 0.94);
    border-right: 1px solid var(--border);
    overflow: hidden;
  }
  .rail-header {
    padding: 16px 14px 12px;
    border-bottom: 1px solid var(--border);
  }
  .brand-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--cyan);
    margin-bottom: 6px;
  }
  .rail-title { font-size: 14px; font-weight: 700; color: var(--text); }
  .rail-sub { font-size: 11px; color: var(--muted); margin-top: 3px; }
  .rail-search {
    width: 100%; margin-top: 10px; padding: 7px 10px;
    border-radius: 8px; border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.9); color: var(--text);
    font-size: 12px; outline: none;
  }
  .rail-search:focus { border-color: var(--cyan); }
  .rail-list {
    flex: 1; overflow-y: auto; padding: 10px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .target-card {
    padding: 10px 11px; border-radius: 10px;
    background: var(--card); border: 1px solid var(--border);
    cursor: pointer; transition: all 0.18s ease;
  }
  .target-card:hover { border-color: rgba(56, 189, 248, 0.5); transform: translateY(-1px); }
  .target-card.active {
    border-color: var(--cyan);
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.16), rgba(16, 185, 129, 0.10));
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.25);
  }
  .tc-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
  .tc-name { font-size: 12.5px; font-weight: 700; color: var(--text); }
  .tc-pdb {
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px; font-weight: 700; padding: 2px 6px;
    border-radius: 5px; background: rgba(56, 189, 248, 0.15); color: var(--cyan);
  }
  .tc-disease { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
  .tc-badges { display: flex; gap: 6px; margin-top: 7px; flex-wrap: wrap; }
  .mini-badge {
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px; padding: 2px 6px; border-radius: 5px;
    background: rgba(148, 163, 184, 0.12); color: #E2E8F0;
  }

  /* Center 3D WebGL Stage */
  .viewport-wrap {
    position: relative; width: 100%; height: 100%; overflow: hidden;
  }
  #mol3d-stage {
    position: absolute; inset: 0; width: 100%; height: 100%;
  }
  .stage-topbar {
    position: absolute; top: 14px; left: 16px; right: 16px; z-index: 20;
    display: flex; justify-content: space-between; align-items: flex-start;
    pointer-events: none; gap: 12px; flex-wrap: wrap;
  }
  .glass-bar {
    pointer-events: auto;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    padding: 7px 10px; border-radius: 12px;
    background: var(--panel); border: 1px solid var(--border);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.42);
  }
  .ctrl-btn {
    border: 1px solid var(--border); background: rgba(30, 41, 59, 0.72);
    color: var(--text); font-size: 11.5px; font-weight: 600;
    padding: 5px 10px; border-radius: 8px; cursor: pointer;
    transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 5px;
  }
  .ctrl-btn:hover { border-color: var(--cyan); background: rgba(56, 189, 248, 0.16); }
  .ctrl-btn.active {
    border-color: var(--cyan); background: rgba(56, 189, 248, 0.24); color: #E0F2FE;
  }
  .ctrl-select {
    border: 1px solid var(--border); background: rgba(15, 23, 42, 0.92);
    color: var(--text); font-size: 11.5px; font-weight: 600;
    padding: 5px 9px; border-radius: 8px; cursor: pointer; outline: none;
  }
  .stage-bottombar {
    position: absolute; bottom: 14px; left: 16px; right: 16px; z-index: 20;
    display: flex; justify-content: space-between; align-items: flex-end;
    pointer-events: none; gap: 10px; flex-wrap: wrap;
  }
  .int-pill {
    pointer-events: auto;
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; border-radius: 999px; font-size: 11px; font-weight: 600;
    background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border);
    color: var(--text); cursor: pointer; transition: all 0.15s ease;
  }
  .int-pill.off { opacity: 0.38; filter: grayscale(0.6); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .offline-banner {
    display: none; position: absolute; inset: 0; z-index: 15;
    align-items: center; justify-content: center; flex-direction: column;
    text-align: center; padding: 32px; color: var(--muted);
  }

  /* Right Glassmorphic Scientific HUD Sidebar */
  .hud-sidebar {
    background: var(--panel);
    border-left: 1px solid var(--border);
    backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
    overflow-y: auto; padding: 16px 16px 22px;
    display: flex; flex-direction: column; gap: 14px;
  }
  .hud-title { font-size: 17px; font-weight: 800; line-height: 1.25; color: var(--text); }
  .hud-tags { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 7px; }
  .tag-pill {
    font-size: 10.5px; font-weight: 700; padding: 3px 8px; border-radius: 6px;
    background: rgba(56, 189, 248, 0.14); color: var(--cyan);
    border: 1px solid rgba(56, 189, 248, 0.25);
  }
  .tag-pill.disease {
    background: rgba(244, 114, 182, 0.13); color: var(--magenta);
    border-color: rgba(244, 114, 182, 0.25);
  }
  .svg-card {
    background: rgba(10, 16, 32, 0.9); border: 1px solid var(--border);
    border-radius: 12px; padding: 8px; text-align: center;
  }
  .svg-card svg { max-width: 100%; height: auto; display: block; margin: 0 auto; }
  .smiles-box {
    margin-top: 7px; padding: 6px 8px; border-radius: 7px;
    background: rgba(30, 41, 59, 0.65); border: 1px solid var(--border);
    font-family: "SF Mono", "JetBrains Mono", monospace;
    font-size: 10px; color: #CBD5E1; word-break: break-all;
    cursor: pointer; text-align: left;
  }
  .smiles-box:hover { border-color: var(--cyan); color: #F8FAFC; }
  .section-label {
    font-size: 10px; font-weight: 800; letter-spacing: 0.11em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 7px;
  }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px;
  }
  .kpi-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 9px 10px; position: relative; overflow: hidden;
  }
  .kpi-card::before {
    content: ""; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px;
    border-radius: 2px; background: var(--accent, var(--cyan));
  }
  .kpi-label { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; }
  .kpi-val {
    font-size: 15px; font-weight: 800; color: var(--text); margin-top: 2px;
    font-family: "SF Mono", "JetBrains Mono", monospace;
  }
  .kpi-sub { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .mpo-bar {
    height: 4px; border-radius: 2px; background: rgba(148, 163, 184, 0.2);
    margin-top: 5px; overflow: hidden;
  }
  .mpo-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #38BDF8, #34D399);
  }
  .int-table {
    width: 100%; border-collapse: collapse; font-size: 11px;
  }
  .int-table th {
    text-align: left; font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); padding: 5px 6px; border-bottom: 1px solid var(--border);
  }
  .int-table td {
    padding: 6px 6px; border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    font-family: "SF Mono", "JetBrains Mono", monospace;
  }
  .int-table tr.int-row { cursor: pointer; transition: background 0.14s ease; }
  .int-table tr.int-row:hover { background: rgba(56, 189, 248, 0.12); }
  .kind-chip {
    display: inline-block; padding: 2px 6px; border-radius: 5px;
    font-size: 9.5px; font-weight: 700; font-family: inherit;
  }
  .notes-box {
    font-size: 11.5px; line-height: 1.48; color: #CBD5E1;
    background: rgba(15, 23, 42, 0.72); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 11px;
  }
  @media (max-width: 1024px) {
    .studio-shell { grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }
    .gallery-rail { display: none; }
    .hud-sidebar { border-left: none; border-top: 1px solid var(--border); }
  }
</style>
</head>
<body>
<div class="studio-shell">
  <aside class="gallery-rail" id="gallery-rail">
    <div class="rail-header">
      <div class="brand-pill">◆ GenEuclidean v3.1 Studio</div>
      <div class="rail-title">__GALLERY_TITLE__</div>
      <div class="rail-sub">__GALLERY_SUBTITLE__</div>
      <input id="rail-search" class="rail-search" type="search"
             placeholder="Filter targets, PDB, disease..." />
    </div>
    <div class="rail-list" id="rail-list"></div>
  </aside>

  <main class="viewport-wrap">
    <div id="mol3d-stage"></div>
    <div class="offline-banner" id="offline-banner">
      <h3 style="color:#F8FAFC;margin-bottom:8px;">
        3Dmol.js WebGL Engine Requires Network Script
      </h3>
      <p style="max-width:420px;font-size:13px;">
        Open with internet access to load 3Dmol.js interactively, or inspect the embedded
        2D structure, CNS/ADMET KPIs, and exact 3D non-covalent interactions in the HUD panel.
      </p>
    </div>

    <div class="stage-topbar">
      <div class="glass-bar">
        <button class="ctrl-btn active" id="btn-surface">Surface</button>
        <button class="ctrl-btn active" id="btn-sidechains">Pocket Sticks</button>
        <button class="ctrl-btn active" id="btn-native" style="display:none;">Native Ghost</button>
        <select class="ctrl-select" id="sel-color-mode" title="Protein Color Mode">
          <option value="attention">Color: 3D Perceiver Attention</option>
          <option value="hydrophobicity">Color: Hydrophobicity</option>
          <option value="chain">Color: Chain / Cartoon</option>
        </select>
      </div>
      <div class="glass-bar">
        <button class="ctrl-btn" id="btn-zoom-site">Binding Site</button>
        <button class="ctrl-btn" id="btn-zoom-all">Full Pocket</button>
        <button class="ctrl-btn" id="btn-reset">Reset</button>
        <button class="ctrl-btn active" id="btn-labels">Å Labels</button>
        <button class="ctrl-btn" id="btn-spin">Spin</button>
        <button class="ctrl-btn" id="btn-snap">Snapshot PNG</button>
      </div>
    </div>

    <div class="stage-bottombar">
      <div class="glass-bar" id="interaction-toggles"></div>
      <div class="glass-bar" style="font-size:11px;color:var(--muted);">
        <span><strong style="color:#34D399;">●</strong> De Novo Hit</span>
        <span id="legend-native" style="display:none;margin-left:8px;">
          <strong style="color:#38BDF8;">●</strong> Native Co-Crystal
        </span>
      </div>
    </div>
  </main>

  <aside class="hud-sidebar" id="hud-sidebar">
    <div>
      <div class="brand-pill">◆ GenEuclidean v3.1 • 3D Complex Inspector</div>
      <div class="hud-title" id="hud-title">—</div>
      <div class="hud-tags" id="hud-tags"></div>
    </div>

    <div class="svg-card">
      <div id="hud-svg-2d"></div>
      <div class="smiles-box" id="hud-smiles" title="Click to copy SMILES"></div>
    </div>

    <div>
      <div class="section-label">Medicinal Chemistry &amp; CNS ADMET Profile</div>
      <div class="kpi-grid" id="hud-kpis"></div>
    </div>

    <div>
      <div class="section-label" id="hud-int-header">3D Non-Covalent Interactions</div>
      <table class="int-table">
        <thead>
          <tr><th>Type</th><th>Residue</th><th>Dist</th><th>Angle</th></tr>
        </thead>
        <tbody id="hud-int-tbody"></tbody>
      </table>
    </div>

    <div class="notes-box" id="hud-notes" style="display:none;"></div>
  </aside>
</div>

<script id="geneuclidean-studio-data" type="application/json">__EMBEDDED_JSON__</script>
<script>
(function() {
  const DATA = JSON.parse(document.getElementById("geneuclidean-studio-data").textContent);
  const SPECS = DATA.specs || [];
  const INTERACTION_STYLE = {
    hbond:        { label: "H-Bond",       short: "HB",    color: "#38BDF8" },
    salt_bridge:  { label: "Salt Bridge",  short: "SB",    color: "#F472B6" },
    pi_stacking:  { label: "π–π Stack",    short: "π–π",   color: "#FBBF24" },
    hydrophobic:  { label: "Hydrophobic",  short: "HYD",   color: "#34D399" },
    cation_pi:    { label: "Cation–π",     short: "CAT-π", color: "#FB923C" },
    halogen_bond: { label: "Halogen",      short: "XB",    color: "#A78BFA" }
  };
  const HYDROPATHY = {
    ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5, MET: 1.9, ALA: 1.8,
    GLY: -0.4, THR: -0.7, SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6,
    HIS: -3.2, GLU: -3.5, GLN: -3.5, ASP: -3.5, ASN: -3.5, LYS: -3.9, ARG: -4.5
  };

  const state = {
    activeIdx: 0,
    showSurface: true,
    showSidechains: true,
    showNative: true,
    showLabels: true,
    spinning: false,
    colorMode: "attention",
    visibleKinds: {
      hbond: true, salt_bridge: true, pi_stacking: true,
      hydrophobic: true, cation_pi: true, halogen_bond: true
    }
  };

  let viewer = null;
  let proteinModel = null;
  let ligandModel = null;
  let nativeModel = null;
  let surfaceObj = null;

  function lerpColor(c1, c2, t) {
    const k = Math.max(0, Math.min(1, t));
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * k);
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * k);
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * k);
    return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
  }

  function attentionColor(score) {
    const s = Math.max(0, Math.min(1, score || 0));
    const stops = [
      [0.00, [30, 27, 75]],
      [0.38, [79, 70, 229]],
      [0.72, [245, 158, 11]],
      [1.00, [253, 224, 71]]
    ];
    for (let i = 0; i < stops.length - 1; i++) {
      if (s <= stops[i + 1][0]) {
        const t = (s - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
        return lerpColor(stops[i][1], stops[i + 1][1], t);
      }
    }
    return "#FDE047";
  }

  function hydroColor(resn) {
    const h = HYDROPATHY[(resn || "").toUpperCase()] ?? 0;
    const t = (h + 4.5) / 9.0;
    return t < 0.5
      ? lerpColor([56, 189, 248], [148, 163, 184], t * 2)
      : lerpColor([148, 163, 184], [245, 158, 11], (t - 0.5) * 2);
  }

  function lookupAttention(atom, attnMap) {
    if (!attnMap) return 0.12;
    const resn = (atom.resn || "").toUpperCase();
    const resi = String(atom.resi);
    const chain = (atom.chain || "A").trim() || "A";
    const full = `${resn}${resi}:${chain}`;
    const shortKey = `${resn}${resi}`;
    if (full in attnMap) return attnMap[full];
    if (shortKey in attnMap) return attnMap[shortKey];
    if (resi in attnMap) return attnMap[resi];
    return 0.12;
  }

  function buildProteinColorFunc(spec) {
    const hasAttn = spec.attention && Object.keys(spec.attention).length > 0;
    return function(atom) {
      if (state.colorMode === "attention" && hasAttn) {
        return attentionColor(lookupAttention(atom, spec.attention));
      }
      if (state.colorMode === "hydrophobicity") {
        return hydroColor(atom.resn);
      }
      const chainPalette = { A: "#475569", B: "#6366F1", C: "#0EA5E9", D: "#14B8A6" };
      return chainPalette[atom.chain] || "#475569";
    };
  }

  function parseResidueSelector(resStr) {
    const parts = resStr.split(":");
    const chain = parts.length > 1 ? parts[1] : undefined;
    const match = parts[0].match(/(\d+)/);
    const resi = match ? parseInt(match[1], 10) : undefined;
    const sel = {};
    if (resi !== undefined) sel.resi = resi;
    if (chain) sel.chain = chain;
    return sel;
  }

  function renderGalleryRail(filterText = "") {
    const list = document.getElementById("rail-list");
    if (!list) return;
    const q = filterText.trim().toLowerCase();
    list.innerHTML = "";
    SPECS.forEach((sp, idx) => {
      const hay = `${sp.title} ${sp.pdb_id} ${sp.disease} ${sp.target_id}`.toLowerCase();
      if (q && !hay.includes(q)) return;
      const card = document.createElement("div");
      card.className = "target-card" + (idx === state.activeIdx ? " active" : "");
      const m = sp.metrics || {};
      const vina = m.vina_kcal ?? m.vina_affinity;
      const mpo = m.cns_mpo;
      const nInt = (sp.profile && sp.profile.interactions) ? sp.profile.interactions.length : 0;
      const vinaTag = vina !== undefined
        ? `<span class="mini-badge">Vina ${Number(vina).toFixed(2)}</span>` : "";
      const mpoTag = mpo !== undefined
        ? `<span class="mini-badge">MPO ${Number(mpo).toFixed(1)}</span>` : "";
      card.innerHTML = `
        <div class="tc-top">
          <span class="tc-name">${sp.title}</span>
          ${sp.pdb_id ? `<span class="tc-pdb">${sp.pdb_id}</span>` : ""}
        </div>
        ${sp.disease ? `<div class="tc-disease">${sp.disease}</div>` : ""}
        <div class="tc-badges">
          ${vinaTag}${mpoTag}
          <span class="mini-badge">${nInt} contacts</span>
        </div>
      `;
      card.addEventListener("click", () => selectSpec(idx));
      list.appendChild(card);
    });
  }

  function renderHud(spec) {
    document.getElementById("hud-title").textContent = spec.title || spec.target_id;
    const tags = document.getElementById("hud-tags");
    tags.innerHTML = "";
    if (spec.pdb_id) {
      tags.innerHTML += `<span class="tag-pill">PDB ${spec.pdb_id}</span>`;
    }
    if (spec.disease) {
      tags.innerHTML += `<span class="tag-pill disease">${spec.disease}</span>`;
    }

    document.getElementById("hud-svg-2d").innerHTML = spec.svg_2d || "";
    const smiBox = document.getElementById("hud-smiles");
    smiBox.textContent = spec.smiles || "—";
    smiBox.onclick = () => {
      if (navigator.clipboard && spec.smiles) {
        navigator.clipboard.writeText(spec.smiles);
        smiBox.style.borderColor = "#34D399";
        setTimeout(() => { smiBox.style.borderColor = ""; }, 900);
      }
    };

    // Build scientific KPI grid
    const m = spec.metrics || {};
    const kpiGrid = document.getElementById("hud-kpis");
    kpiGrid.innerHTML = "";
    const addKpi = (label, value, sub, accent, extraHtml = "") => {
      const div = document.createElement("div");
      div.className = "kpi-card";
      div.style.setProperty("--accent", accent);
      div.innerHTML =
        `<div class="kpi-label">${label}</div>` +
        `<div class="kpi-val">${value}</div>` +
        `<div class="kpi-sub">${sub}</div>${extraHtml}`;
      kpiGrid.appendChild(div);
    };

    const vina = m.vina_kcal ?? m.vina_affinity;
    const natVina = m.native_vina_kcal ?? m.native_vina;
    if (vina !== undefined) {
      const sub = natVina !== undefined
        ? `Native ${Number(natVina).toFixed(2)} (Δ${(Number(vina) - Number(natVina)).toFixed(2)})`
        : "AutoDock Vina kcal/mol";
      addKpi(
        "Vina ΔG", `${Number(vina).toFixed(2)}`, sub,
        Number(vina) <= -8.0 ? "#34D399" : "#38BDF8"
      );
    }
    if (m.cns_mpo !== undefined) {
      const pct = Math.max(0, Math.min(100, (Number(m.cns_mpo) / 6.0) * 100));
      addKpi(
        "Pfizer CNS MPO", `${Number(m.cns_mpo).toFixed(2)} / 6`, "CNS desirability",
        Number(m.cns_mpo) >= 4.5 ? "#34D399" : "#FBBF24",
        `<div class="mpo-bar"><div class="mpo-fill" style="width:${pct}%"></div></div>`
      );
    }
    if (m.bbb_prob !== undefined) {
      addKpi(
        "P(BBB+)", `${Math.round(Number(m.bbb_prob) * 100)}%`, "Brain penetration",
        Number(m.bbb_prob) >= 0.7 ? "#34D399" : "#FBBF24"
      );
    }
    if (m.pgp_minus_prob !== undefined) {
      addKpi(
        "P(Pgp−)", `${Math.round(Number(m.pgp_minus_prob) * 100)}%`, "Low P-gp efflux",
        Number(m.pgp_minus_prob) >= 0.65 ? "#34D399" : "#FBBF24"
      );
    }
    if (m.herg_prob !== undefined) {
      addKpi(
        "P(hERG)", `${Math.round(Number(m.herg_prob) * 100)}%`, "Cardiotox risk",
        Number(m.herg_prob) <= 0.30 ? "#34D399" : "#FB7185"
      );
    }
    if (m.qed !== undefined) {
      addKpi(
        "QED", Number(m.qed).toFixed(3), "Drug-likeness",
        Number(m.qed) >= 0.65 ? "#34D399" : "#38BDF8"
      );
    }
    if (m.sa !== undefined || m.sa_score !== undefined) {
      const sa = m.sa ?? m.sa_score;
      addKpi(
        "SA Score", Number(sa).toFixed(2), "Synthetic access.",
        Number(sa) <= 3.2 ? "#34D399" : "#FBBF24"
      );
    }
    if (m.posebusters_pass !== undefined || m.pb_pass !== undefined) {
      const ok = Boolean(m.posebusters_pass ?? m.pb_pass);
      addKpi(
        "PoseBusters", ok ? "✓ PASS" : "✗ FAIL", "3D geometry audit",
        ok ? "#34D399" : "#FB7185"
      );
    }
    if (m.plif_tanimoto !== undefined) {
      addKpi("PLIF Tanimoto", Number(m.plif_tanimoto).toFixed(2), "3D contact recovery", "#A78BFA");
    }
    if (m.rmsd_angstrom !== undefined) {
      addKpi(
        "Heavy RMSD", `${Number(m.rmsd_angstrom).toFixed(2)} Å`, "Symmetry-corrected",
        Number(m.rmsd_angstrom) <= 2.0 ? "#34D399" : "#FBBF24"
      );
    }

    // Interaction toggles & clickable table
    const interactions = spec.profile?.interactions || [];
    const counts = spec.profile?.counts_by_type || {};
    const hdrEl = document.getElementById("hud-int-header");
    hdrEl.textContent = `3D Non-Covalent Interactions (${interactions.length})`;

    const toggleBar = document.getElementById("interaction-toggles");
    toggleBar.innerHTML = "";
    Object.keys(INTERACTION_STYLE).forEach(kind => {
      const st = INTERACTION_STYLE[kind];
      const cnt = counts[kind] || 0;
      const btn = document.createElement("button");
      btn.className = "int-pill" + (state.visibleKinds[kind] ? "" : " off");
      btn.innerHTML =
        `<span class="dot" style="background:${st.color}"></span>${st.label} (${cnt})`;
      btn.onclick = () => {
        state.visibleKinds[kind] = !state.visibleKinds[kind];
        renderHud(spec);
        redrawInteractionsAndLabels(spec);
      };
      toggleBar.appendChild(btn);
    });

    const tbody = document.getElementById("hud-int-tbody");
    tbody.innerHTML = "";
    interactions.forEach(rec => {
      const st = INTERACTION_STYLE[rec.kind] || INTERACTION_STYLE.hbond;
      const tr = document.createElement("tr");
      tr.className = "int-row";
      const hasAng = rec.angle_deg !== null && rec.angle_deg !== undefined;
      const angTxt = hasAng ? `${Number(rec.angle_deg).toFixed(0)}°` : "—";
      const chipStyle = `background:${st.color}22;color:${st.color};`;
      tr.innerHTML = `
        <td><span class="kind-chip" style="${chipStyle}">${st.short}</span></td>
        <td>${rec.protein_residue} <span style="color:#64748B">${rec.protein_atom}</span></td>
        <td>${Number(rec.distance_angstrom).toFixed(2)} Å</td>
        <td style="color:#94A3B8">${angTxt}</td>
      `;
      tr.onclick = () => {
        if (!viewer) return;
        const mx = 0.5 * (rec.protein_xyz[0] + rec.ligand_xyz[0]);
        const my = 0.5 * (rec.protein_xyz[1] + rec.ligand_xyz[1]);
        const mz = 0.5 * (rec.protein_xyz[2] + rec.ligand_xyz[2]);
        viewer.zoomTo({ model: 0, ...parseResidueSelector(rec.protein_residue) }, 420);
        viewer.center({ x: mx, y: my, z: mz }, 380);
      };
      tbody.appendChild(tr);
    });

    const notesEl = document.getElementById("hud-notes");
    if (spec.notes) {
      notesEl.style.display = "block";
      notesEl.textContent = spec.notes;
    } else {
      notesEl.style.display = "none";
    }
  }

  function applyProteinStyling(spec) {
    if (!viewer || !proteinModel) return;
    const colorfn = buildProteinColorFunc(spec);
    proteinModel.setStyle({}, {
      cartoon: { colorfunc: colorfn, opacity: 0.78, thickness: 0.36 }
    });

    if (state.showSidechains) {
      const keyRes = (spec.profile && spec.profile.key_residues) ? spec.profile.key_residues : [];
      keyRes.slice(0, 14).forEach(resStr => {
        const sel = parseResidueSelector(resStr);
        proteinModel.addStyle(sel, {
          stick: { radius: 0.12, colorscheme: "Jmol", opacity: 0.90 }
        });
      });
    }

    if (surfaceObj !== null) {
      try { viewer.removeSurface(surfaceObj); } catch (e) {}
      surfaceObj = null;
    }
    if (state.showSurface && window.$3Dmol) {
      surfaceObj = viewer.addSurface(
        $3Dmol.SurfaceType.SAS,
        { opacity: 0.24, colorfunc: colorfn },
        { model: proteinModel }
      );
    }
  }

  function applyLigandStyling(spec) {
    if (!viewer) return;
    if (ligandModel) {
      ligandModel.setStyle({}, {
        stick: {
          radius: 0.18,
          colorscheme: {
            prop: "elem",
            map: {
              C: "#10B981", N: "#3B82F6", O: "#EF4444", S: "#FACC15",
              F: "#4ADE80", Cl: "#22C55E", Br: "#A855F7", I: "#C084FC",
              P: "#F97316", H: "#E2E8F0"
            }
          }
        },
        sphere: {
          scale: 0.27,
          colorscheme: {
            prop: "elem",
            map: {
              C: "#10B981", N: "#3B82F6", O: "#EF4444", S: "#FACC15",
              F: "#4ADE80", Cl: "#22C55E", Br: "#A855F7", I: "#C084FC",
              P: "#F97316", H: "#E2E8F0"
            }
          }
        }
      });
    }
    if (nativeModel) {
      if (state.showNative) {
        nativeModel.setStyle({}, {
          stick: { radius: 0.13, color: "#38BDF8", opacity: 0.45 }
        });
      } else {
        nativeModel.setStyle({}, {});
      }
    }
  }

  function redrawInteractionsAndLabels(spec) {
    if (!viewer) return;
    viewer.removeAllShapes();
    viewer.removeAllLabels();

    const interactions = spec.profile?.interactions || [];
    const labeledRes = new Set();

    interactions.forEach(rec => {
      if (!state.visibleKinds[rec.kind]) return;
      const st = INTERACTION_STYLE[rec.kind] || INTERACTION_STYLE.hbond;
      const p = { x: rec.protein_xyz[0], y: rec.protein_xyz[1], z: rec.protein_xyz[2] };
      const l = { x: rec.ligand_xyz[0], y: rec.ligand_xyz[1], z: rec.ligand_xyz[2] };

      viewer.addCylinder({
        start: p,
        end: l,
        radius: rec.kind === "hydrophobic" ? 0.055 : 0.075,
        dashed: true,
        dashLength: 0.24,
        gapLength: 0.16,
        color: st.color,
        opacity: rec.kind === "hydrophobic" ? 0.68 : 0.96
      });

      if (state.showLabels && rec.kind !== "hydrophobic") {
        viewer.addLabel(`${Number(rec.distance_angstrom).toFixed(1)} Å`, {
          position: { x: 0.5 * (p.x + l.x), y: 0.5 * (p.y + l.y), z: 0.5 * (p.z + l.z) },
          fontSize: 10,
          fontColor: st.color,
          backgroundColor: "rgba(5, 8, 20, 0.80)",
          borderColor: st.color,
          borderThickness: 1,
          backgroundOpacity: 0.85,
          inFront: true
        });
      }

      const shouldLabelRes =
        state.showSidechains && !labeledRes.has(rec.protein_residue) && rec.kind !== "hydrophobic";
      if (shouldLabelRes) {
        labeledRes.add(rec.protein_residue);
        viewer.addLabel(rec.protein_residue.replace(/:A$/, ""), {
          position: p,
          fontSize: 11,
          fontColor: "#F8FAFC",
          backgroundColor: "rgba(15, 23, 42, 0.86)",
          borderColor: st.color,
          borderThickness: 1,
          backgroundOpacity: 0.88,
          inFront: true
        });
      }
    });
    viewer.render();
  }

  function loadComplexIntoViewer(spec) {
    if (!window.$3Dmol) {
      document.getElementById("offline-banner").style.display = "flex";
      return;
    }
    const stage = document.getElementById("mol3d-stage");
    if (!viewer) {
      viewer = $3Dmol.createViewer(stage, {
        backgroundColor: "#050814",
        antialias: true,
        cartoonQuality: 8
      });
    }
    viewer.clear();
    proteinModel = null;
    ligandModel = null;
    nativeModel = null;
    surfaceObj = null;

    if (spec.pdb) {
      proteinModel = viewer.addModel(spec.pdb, "pdb");
    }
    if (spec.ligand_sdf) {
      ligandModel = viewer.addModel(spec.ligand_sdf, "sdf");
    }
    if (spec.native_sdf) {
      nativeModel = viewer.addModel(spec.native_sdf, "sdf");
    }

    const hasNative = Boolean(spec.native_sdf);
    document.getElementById("btn-native").style.display = hasNative ? "inline-flex" : "none";
    document.getElementById("legend-native").style.display = hasNative ? "inline" : "none";

    // Auto-select chain mode if no attention map was supplied
    const hasAttn = spec.attention && Object.keys(spec.attention).length > 0;
    const selColor = document.getElementById("sel-color-mode");
    if (!hasAttn && state.colorMode === "attention") {
      state.colorMode = "hydrophobicity";
      selColor.value = "hydrophobicity";
    }

    applyProteinStyling(spec);
    applyLigandStyling(spec);
    redrawInteractionsAndLabels(spec);

    if (ligandModel) {
      viewer.zoomTo({ model: ligandModel }, 0);
      viewer.zoom(0.84, 0);
    } else {
      viewer.zoomTo({}, 0);
    }
    viewer.render();
    if (state.spinning) viewer.spin("y", 0.35);
  }

  function selectSpec(idx) {
    if (idx < 0 || idx >= SPECS.length) return;
    state.activeIdx = idx;
    const spec = SPECS[idx];
    renderGalleryRail(document.getElementById("rail-search")?.value || "");
    renderHud(spec);
    loadComplexIntoViewer(spec);
  }

  function bindControls() {
    const cur = () => SPECS[state.activeIdx];

    document.getElementById("btn-surface").onclick = function() {
      state.showSurface = !state.showSurface;
      this.classList.toggle("active", state.showSurface);
      applyProteinStyling(cur());
      if (viewer) viewer.render();
    };
    document.getElementById("btn-sidechains").onclick = function() {
      state.showSidechains = !state.showSidechains;
      this.classList.toggle("active", state.showSidechains);
      applyProteinStyling(cur());
      redrawInteractionsAndLabels(cur());
    };
    document.getElementById("btn-native").onclick = function() {
      state.showNative = !state.showNative;
      this.classList.toggle("active", state.showNative);
      applyLigandStyling(cur());
      if (viewer) viewer.render();
    };
    document.getElementById("sel-color-mode").onchange = function() {
      state.colorMode = this.value;
      applyProteinStyling(cur());
      if (viewer) viewer.render();
    };
    document.getElementById("btn-labels").onclick = function() {
      state.showLabels = !state.showLabels;
      this.classList.toggle("active", state.showLabels);
      redrawInteractionsAndLabels(cur());
    };
    document.getElementById("btn-zoom-site").onclick = () => {
      if (!viewer) return;
      if (ligandModel) {
        viewer.zoomTo({ model: ligandModel }, 380);
        viewer.zoom(0.86, 380);
      }
    };
    document.getElementById("btn-zoom-all").onclick = () => {
      if (viewer) viewer.zoomTo({}, 420);
    };
    document.getElementById("btn-reset").onclick = () => {
      if (!viewer) return;
      state.spinning = false;
      document.getElementById("btn-spin").classList.remove("active");
      viewer.spin(false);
      loadComplexIntoViewer(cur());
    };
    document.getElementById("btn-spin").onclick = function() {
      if (!viewer) return;
      state.spinning = !state.spinning;
      this.classList.toggle("active", state.spinning);
      viewer.spin(state.spinning ? "y" : false, 0.35);
    };
    document.getElementById("btn-snap").onclick = () => {
      if (!viewer) return;
      const uri = viewer.pngURI();
      const a = document.createElement("a");
      a.href = uri;
      const slug = (cur().target_id || "geneuclidean_complex").replace(/[^a-zA-Z0-9_-]/g, "_");
      a.download = `${slug}_3d.png`;
      a.click();
    };
    const searchEl = document.getElementById("rail-search");
    if (searchEl) {
      searchEl.addEventListener("input", e => renderGalleryRail(e.target.value));
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    bindControls();
    if (SPECS.length > 0) {
      selectSpec(0);
    }
  });
})();
</script>
</body>
</html>
"""


def render_interactive_3d_gallery_html(
    specs: Sequence[ComplexShowcaseSpec],
    out_path: str | Path,
    *,
    gallery_title: str = "GenEuclidean v3.1 — 3D Molecular Studio",
    subtitle: str = "Interactive Protein-Ligand WebGL & PLIF Inspector",
) -> Path:
    """Generate a self-contained dark-mode 3Dmol.js WebGL multi-target gallery HTML dashboard.

    Parameters
    ----------
    specs:
        Non-empty sequence of :class:`ComplexShowcaseSpec` entries (e.g. Tourette/psychiatric
        CNS targets and CASF-2016 showcase hits). When a single spec is provided, the left
        gallery switcher rail is automatically hidden for maximum viewport space.
    out_path:
        Destination ``.html`` file path.
    gallery_title:
        Title displayed in the browser tab and gallery rail header.
    subtitle:
        Subtitle displayed beneath the gallery rail title.

    Returns
    -------
    Path
        Path to the written self-contained HTML5 WebGL viewer.
    """
    if not specs:
        raise ValueError("specs must contain at least one ComplexShowcaseSpec.")

    serialized_specs = [_serialize_showcase_spec(s) for s in specs]
    is_gallery = len(serialized_specs) > 1
    payload = {
        "studio_version": "3.1",
        "gallery_title": gallery_title,
        "specs": serialized_specs,
    }

    html_out = (
        _INTERACTIVE_HTML_TEMPLATE.replace("__PAGE_TITLE__", html_lib.escape(gallery_title))
        .replace("__GALLERY_TITLE__", html_lib.escape(gallery_title))
        .replace("__GALLERY_SUBTITLE__", html_lib.escape(subtitle))
        .replace(
            "__GRID_COLS__",
            "268px minmax(0, 1fr) 360px" if is_gallery else "minmax(0, 1fr) 370px",
        )
        .replace("__RAIL_DISPLAY__", "flex" if is_gallery else "none")
        .replace("__EMBEDDED_JSON__", _safe_inline_json(payload))
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    return out


def render_interactive_3d_html(
    pocket_pdb_path: str | Path,
    ligand_mol: Chem.Mol,
    out_path: str | Path,
    *,
    native_ligand_mol: Chem.Mol | None = None,
    interactions: InteractionProfile | None = None,
    title: str = "3D Protein-Ligand Complex",
    pdb_id: str = "",
    disease: str = "",
    smiles: str = "",
    metrics: Mapping[str, Any] | None = None,
    residue_attention: Mapping[str, float] | None = None,
    notes: str = "",
) -> Path:
    """Generate a self-contained dark-mode 3Dmol.js WebGL interactive HTML viewer for one complex.

    Opening the output ``.html`` file in Chrome, Safari, or Firefox renders the protein
    pocket ribbon, surface, attention hotspots, CPK ligand, optional native co-crystal
    ghost overlay, dashed 3D non-covalent interaction cylinders, and glassmorphic HUD
    with zero backend server required.
    """
    spec = ComplexShowcaseSpec(
        target_id=pdb_id or title.lower().replace(" ", "_")[:24],
        title=title,
        pdb_id=pdb_id,
        disease=disease,
        pocket_pdb_path=pocket_pdb_path,
        ligand_mol=ligand_mol,
        native_ligand_mol=native_ligand_mol,
        smiles=smiles,
        interactions=interactions,
        residue_attention=residue_attention,
        metrics=dict(metrics) if metrics else {},
        notes=notes,
    )
    return render_interactive_3d_gallery_html(
        [spec],
        out_path=out_path,
        gallery_title=f"{title} — GenEuclidean v3.1 3D Studio",
        subtitle=disease or "3D Protein-Ligand WebGL Inspector",
    )
