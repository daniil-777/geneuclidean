"""Unit and live-HTTP tests for the Studio 3D WebGL2 interaction viewer & trajectory player.

Covers:
1. Shipped ``VIEWER_ASSET`` existence, balanced HTML5 fragment parsing, WebGL2 ``gl_FragDepth``
   sphere impostors, ``gl.depthMask(false)`` halo fix, ``.curtain[hidden]`` visibility rule,
   live ``U (frame)`` readout, ``pdb_id|style`` surface cache key, and linear interpolation.
2. ``render_viewer_document()`` HTML5 document wrapper, custom ``scenes`` & ``scene_base``
   injection, script-tag breakout escaping, and trailing-slash validation.
3. ``build_interaction_scene()`` and ``write_viewer_bundle()`` on a synthetic 3-residue
   pocket PDB + 3D RDKit ligand, producing ``interaction_viewer.html``,
   ``scenes/<pdb_id>.json`` (``geneuclidean.biophysics.web_twin_scene.v1``), and
   ``scenes/<pdb_id>.traj.json`` (from both ``BindingTrajectory`` and serialized dicts).
4. ``write_viewer_bundle()`` error guards on empty ``complexes``, unknown trajectory keys,
   and ligand-atom count mismatches between trajectory frames and scene.
5. ``resolve_viewer_asset()`` serving expected routes while refusing ``../../../etc/passwd``,
   ``scenes/../../../etc/passwd``, URL-encoded / backslash / null-byte / symlink escapes,
   and non-``.json`` files inside ``bundle_dir``.
6. Live HTTP ``GET /viewer``, ``GET /viewer/scenes/<id>.json``,
   ``GET /viewer/scenes/<id>.traj.json``, and ``GET /viewer/../../../etc/passwd`` (``404``)
   via ``create_studio_server(..., viewer_bundle_dir=...)``.
"""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from geneuclidean.biophysics.binding_dynamics import (
    BindingDynamicsConfig,
    make_dynamics_hamiltonian,
    simulate_binding,
)
from geneuclidean.biophysics.forcefield import (
    parameterize_ligand_biophysics,
    parameterize_pocket_biophysics,
)
from geneuclidean.studio import (
    VIEWER_ASSET,
    VIEWER_SCENE_SCHEMA,
    build_interaction_scene,
    create_studio_server,
    render_viewer_document,
    resolve_viewer_asset,
    write_viewer_bundle,
)

_HTML5_VOID_ELEMENTS: frozenset[str] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

_SYNTHETIC_3RES_POCKET_PDB = """\
ATOM      1  N   ALA A   1      -4.000   3.200   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      -4.000   1.800   0.600  1.00  0.00           C
ATOM      3  C   ALA A   1      -2.700   1.100   0.200  1.00  0.00           C
ATOM      4  O   ALA A   1      -2.600  -0.100   0.500  1.00  0.00           O
ATOM      5  CB  ALA A   1      -5.200   1.000   0.100  1.00  0.00           C
ATOM      6  N   SER A   2       4.000   3.200   0.000  1.00  0.00           N
ATOM      7  CA  SER A   2       4.000   1.800  -0.600  1.00  0.00           C
ATOM      8  C   SER A   2       2.700   1.100  -0.200  1.00  0.00           C
ATOM      9  O   SER A   2       2.600  -0.100  -0.500  1.00  0.00           O
ATOM     10  OG  SER A   2       5.200   1.000  -0.100  1.00  0.00           O
ATOM     11  N   PHE A   3       0.000  -4.200   1.000  1.00  0.00           N
ATOM     12  CA  PHE A   3       0.000  -2.800   1.400  1.00  0.00           C
ATOM     13  C   PHE A   3       1.300  -2.100   1.000  1.00  0.00           C
ATOM     14  O   PHE A   3       1.400  -0.900   1.300  1.00  0.00           O
ATOM     15  CB  PHE A   3      -1.200  -2.000   0.900  1.00  0.00           C
ATOM     16  CG  PHE A   3      -1.400  -2.000  -0.580  1.00  0.00           C
ATOM     17  CD1 PHE A   3      -0.700  -1.150  -1.420  1.00  0.00           C
ATOM     18  CD2 PHE A   3      -2.300  -2.860  -1.150  1.00  0.00           C
ATOM     19  CE1 PHE A   3      -0.890  -1.150  -2.790  1.00  0.00           C
ATOM     20  CE2 PHE A   3      -2.500  -2.870  -2.520  1.00  0.00           C
ATOM     21  CZ  PHE A   3      -1.790  -2.020  -3.340  1.00  0.00           C
END
"""


class _StrictHTMLInspector(HTMLParser):
    """Lightweight HTML5 structure inspector capturing tags, doctypes, and JSON config."""

    def __init__(self, *, allow_implicit_head_body: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.allow_implicit_head_body = allow_implicit_head_body
        self.doctypes: list[str] = []
        self.open_stack: list[str] = []
        self.seen_tags: list[str] = []
        self.viewer_config_raw: str | None = None
        self._capturing_config = False
        self._config_chunks: list[str] = []

    def handle_decl(self, decl: str) -> None:
        self.doctypes.append(decl.strip().lower())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        self.seen_tags.append(lower)
        if (
            self.allow_implicit_head_body
            and lower == "div"
            and self.open_stack
            and self.open_stack[-1] == "head"
        ):
            # HTML5 implicitly closes <head> and opens <body> on the first flow element.
            self.open_stack.pop()
            self.open_stack.append("body")
        if lower not in _HTML5_VOID_ELEMENTS:
            self.open_stack.append(lower)
        attr_map = dict(attrs)
        if lower == "script" and attr_map.get("id") == "viewer-config":
            self._capturing_config = True
            self._config_chunks = []

    def handle_data(self, data: str) -> None:
        if self._capturing_config:
            self._config_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower == "script" and self._capturing_config:
            self._capturing_config = False
            self.viewer_config_raw = "".join(self._config_chunks).strip()
        if lower in _HTML5_VOID_ELEMENTS:
            return
        assert self.open_stack, f"Unexpected closing </{lower}> with empty stack"
        opened = self.open_stack.pop()
        assert opened == lower, f"Mismatched closing </{lower}> for opened <{opened}>"


@pytest.fixture(scope="module")
def synthetic_complex_files(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Chem.Mol]:
    """Build a deterministic 3-residue pocket PDB and 3D ligand SDF in a temp directory."""
    work = tmp_path_factory.mktemp("viewer_fixture")
    pocket_pdb = work / "pocket_3res.pdb"
    pocket_pdb.write_text(_SYNTHETIC_3RES_POCKET_PDB, encoding="utf-8")

    base_mol = Chem.MolFromSmiles("c1ccccc1CO")
    assert base_mol is not None
    mol = Chem.AddHs(base_mol)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    rdForceFieldHelpers.MMFFOptimizeMolecule(mol)

    ligand_sdf = work / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()

    return pocket_pdb, ligand_sdf, mol


# ---------------------------------------------------------------------------
# 1. Shipped Viewer Asset & Browser Defect Regression Checks (§8)
# ---------------------------------------------------------------------------


def test_viewer_asset_exists_parses_and_retains_webgl2_browser_fixes() -> None:
    """Shipped VIEWER_ASSET parses cleanly and preserves all WebGL2 & §8 browser fixes."""
    assert VIEWER_ASSET.is_file(), f"Missing viewer asset at {VIEWER_ASSET}"
    raw = VIEWER_ASSET.read_text(encoding="utf-8")
    assert len(raw) > 20_000

    inspector = _StrictHTMLInspector()
    inspector.feed(raw)
    inspector.close()
    assert inspector.open_stack == [], f"Unclosed tags in VIEWER_ASSET: {inspector.open_stack}"
    assert "title" in inspector.seen_tags
    assert "canvas" in inspector.seen_tags
    assert inspector.viewer_config_raw is not None
    cfg = json.loads(inspector.viewer_config_raw)
    assert cfg["sceneBase"] == "scenes/"
    assert isinstance(cfg["scenes"], list) and len(cfg["scenes"]) >= 1

    # Real WebGL2 sphere impostors + tube & surface shaders (never 2D canvas fallback)
    assert 'getContext("webgl2"' in raw
    assert 'getContext("2d")' not in raw
    assert "#version 300 es" in raw
    assert "gl_FragDepth" in raw

    # §8 Fix 1: Silhouette halo drawn with depth writes disabled so ligand atoms aren't occluded
    halo_idx = raw.index("spheres(B.halo")
    mask_off_idx = raw.rindex("gl.depthMask(false);", 0, halo_idx)
    mask_on_idx = raw.index("gl.depthMask(true);", halo_idx)
    lig_idx = raw.index("spheres(B.ligAtoms", mask_on_idx)
    assert mask_off_idx < halo_idx < mask_on_idx < lig_idx

    # §8 Fix 2: .curtain[hidden] { display: none; } overrides .curtain { display: grid }
    assert ".curtain[hidden]" in raw
    assert "display: none" in raw

    # §8 Fix 3: Live trajectory readout displays frame potential labeled "U (frame)"
    assert "U (frame)" in raw

    # §8 Player invariants: pdb_id|style surface cache key (no ligand coords) + lerp
    assert "${scene.metadata && scene.metadata.pdb_id}|${ui.surf}" in raw
    assert "p[0] + (q[0] - p[0]) * w" in raw


# ---------------------------------------------------------------------------
# 2. Standalone HTML Document Rendering & Config Injection
# ---------------------------------------------------------------------------


def test_render_viewer_document_structure_config_injection_and_validation() -> None:
    """render_viewer_document wraps HTML5 skeleton, injects config, and validates scene_base."""
    doc = render_viewer_document()
    assert doc.lower().startswith("<!doctype html>")
    assert "<title>Pocket Interaction Studio</title>" in doc
    assert "</body>" in doc
    assert "</html>" in doc

    inspector = _StrictHTMLInspector(allow_implicit_head_body=True)
    inspector.feed(doc)
    inspector.close()
    assert inspector.doctypes == ["doctype html"]
    assert inspector.open_stack == []

    custom_scenes = [
        {"id": "3syn", "label": "3SYN", "note": "synthetic · </script> safe"},
        {"id": "4tst", "label": "4TST", "note": "control"},
    ]
    injected = render_viewer_document(scene_base="custom_scenes/", scenes=custom_scenes)
    inj_inspector = _StrictHTMLInspector(allow_implicit_head_body=True)
    inj_inspector.feed(injected)
    inj_inspector.close()
    assert inj_inspector.open_stack == []
    assert inj_inspector.viewer_config_raw is not None
    parsed_cfg = json.loads(inj_inspector.viewer_config_raw)
    assert parsed_cfg["sceneBase"] == "custom_scenes/"
    assert parsed_cfg["scenes"] == custom_scenes

    # Custom scene_base alone updates sceneBase while preserving shipped scenes list
    base_only = render_viewer_document(scene_base="/viewer/scenes/")
    base_inspector = _StrictHTMLInspector(allow_implicit_head_body=True)
    base_inspector.feed(base_only)
    base_inspector.close()
    assert base_inspector.viewer_config_raw is not None
    assert json.loads(base_inspector.viewer_config_raw)["sceneBase"] == "/viewer/scenes/"

    with pytest.raises(ValueError, match="scene_base must end in '/'"):
        render_viewer_document(scene_base="scenes")
    with pytest.raises(ValueError, match="scene_base must end in '/'"):
        render_viewer_document(scene_base="")


# ---------------------------------------------------------------------------
# 3. Scene Construction & Bundle Writer (Static + Trajectory + Validation)
# ---------------------------------------------------------------------------


def test_build_interaction_scene_and_validation(
    synthetic_complex_files: tuple[Path, Path, Chem.Mol],
    tmp_path: Path,
) -> None:
    """build_interaction_scene exports web_twin_scene.v1 from SDF or Mol and rejects 2D inputs."""
    pocket_pdb, ligand_sdf, mol = synthetic_complex_files

    scene_from_mol = build_interaction_scene(
        pocket_pdb,
        mol,
        pdb_id="3syn",
        include_esp_samples=True,
        esp_grid_spacing=3.0,
    )
    assert scene_from_mol["schema_version"] == VIEWER_SCENE_SCHEMA
    assert scene_from_mol["metadata"]["pdb_id"] == "3syn"
    assert len(scene_from_mol["ligand_atoms"]) == mol.GetNumAtoms()
    assert len(scene_from_mol["pocket_atoms"]) == 21
    assert "energy_summary" in scene_from_mol
    assert len(scene_from_mol["esp_field_samples"]) > 0

    scene_no_esp = build_interaction_scene(
        pocket_pdb,
        ligand_sdf,
        pdb_id="3syn_sdf",
        include_esp_samples=False,
    )
    assert scene_no_esp["schema_version"] == VIEWER_SCENE_SCHEMA
    assert len(scene_no_esp["ligand_atoms"]) == mol.GetNumAtoms()
    assert scene_no_esp["esp_field_samples"] == []

    # 0-conformer RDKit Mol must raise ValueError
    flat_mol = Chem.MolFromSmiles("CCO")
    assert flat_mol is not None
    with pytest.raises(ValueError, match="3D conformer"):
        build_interaction_scene(pocket_pdb, flat_mol)

    # Corrupt SDF path must raise ValueError
    bad_sdf = tmp_path / "corrupt.sdf"
    bad_sdf.write_text("not a valid molblock\n$$$$", encoding="utf-8")
    with pytest.raises(ValueError, match="could not parse"):
        build_interaction_scene(pocket_pdb, bad_sdf)


def test_write_viewer_bundle_writes_page_scenes_and_trajectories(
    synthetic_complex_files: tuple[Path, Path, Chem.Mol],
    tmp_path: Path,
) -> None:
    """write_viewer_bundle writes HTML, web_twin_scene.v1 JSON, and .traj.json files."""
    pocket_pdb, ligand_sdf, mol = synthetic_complex_files

    lig_params = parameterize_ligand_biophysics(mol)
    rec_params = parameterize_pocket_biophysics(str(pocket_pdb))
    traj = simulate_binding(
        lig_params,
        rec_params,
        make_dynamics_hamiltonian(),
        config=BindingDynamicsConfig(
            release_steps=6,
            record_every=3,
            launch_distance_a=1.5,
            stage_max_steps=6,
        ),
        backend="python",
    )

    bundle_dir = tmp_path / "bundle"
    page_path = write_viewer_bundle(
        bundle_dir,
        complexes={
            "3syn": (pocket_pdb, ligand_sdf),
            "3mol": (pocket_pdb, mol),
        },
        notes={"3syn": "synthetic 3-res pocket · trajectory"},
        include_esp_samples=False,
        trajectories={"3syn": traj},
    )

    assert page_path == bundle_dir / "interaction_viewer.html"
    assert page_path.is_file()
    html_text = page_path.read_text(encoding="utf-8")
    assert "3SYN" in html_text
    assert "3MOL" in html_text
    assert "synthetic 3-res pocket" in html_text

    scene_3syn_path = bundle_dir / "scenes" / "3syn.json"
    scene_3mol_path = bundle_dir / "scenes" / "3mol.json"
    traj_3syn_path = bundle_dir / "scenes" / "3syn.traj.json"
    traj_3mol_path = bundle_dir / "scenes" / "3mol.traj.json"

    assert scene_3syn_path.is_file()
    assert scene_3mol_path.is_file()
    assert traj_3syn_path.is_file()
    assert not traj_3mol_path.exists()

    scene_data = json.loads(scene_3syn_path.read_text(encoding="utf-8"))
    traj_data = json.loads(traj_3syn_path.read_text(encoding="utf-8"))
    assert scene_data["schema_version"] == VIEWER_SCENE_SCHEMA
    assert traj_data["schema_version"] == "geneuclidean.biophysics.binding_trajectory.v1"
    assert len(traj_data["frames"]) >= 2
    assert len(traj_data["frames"][0]["coords"]) == len(scene_data["ligand_atoms"])


def test_write_viewer_bundle_rejects_empty_unknown_and_mismatched_trajectories(
    synthetic_complex_files: tuple[Path, Path, Chem.Mol],
    tmp_path: Path,
) -> None:
    """write_viewer_bundle raises ValueError on empty complexes, unknown keys, or atom mismatch."""
    pocket_pdb, ligand_sdf, mol = synthetic_complex_files

    with pytest.raises(ValueError, match="at least one complex"):
        write_viewer_bundle(tmp_path / "empty", {})

    with pytest.raises(ValueError, match="not supplied"):
        write_viewer_bundle(
            tmp_path / "unknown_traj",
            {"3syn": (pocket_pdb, ligand_sdf)},
            include_esp_samples=False,
            trajectories={"9zzz": {"frames": []}},
        )

    wrong_atom_count = mol.GetNumAtoms() - 2
    bad_traj_dict = {
        "schema_version": "geneuclidean.biophysics.binding_trajectory.v1",
        "frames": [
            {
                "step": 0,
                "time_ps": 0.0,
                "coords": [[0.0, 0.0, 0.0]] * wrong_atom_count,
                "u_total": -5.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="ligand atoms"):
        write_viewer_bundle(
            tmp_path / "mismatch_traj",
            {"3syn": (pocket_pdb, ligand_sdf)},
            include_esp_samples=False,
            trajectories={"3syn": bad_traj_dict},
        )


# ---------------------------------------------------------------------------
# 4. Asset Resolver Containment & Traversal Guards + Live Studio HTTP Bridge
# ---------------------------------------------------------------------------


def test_resolve_viewer_asset_serves_valid_assets_and_blocks_all_escapes(
    synthetic_complex_files: tuple[Path, Path, Chem.Mol],
    tmp_path: Path,
) -> None:
    """resolve_viewer_asset serves HTML/JSON inside bundle_dir and blocks traversal & symlinks."""
    pocket_pdb, ligand_sdf, _ = synthetic_complex_files
    bundle_dir = tmp_path / "served_bundle"
    write_viewer_bundle(
        bundle_dir,
        {"3syn": (pocket_pdb, ligand_sdf)},
        include_esp_samples=False,
    )

    # Plant non-JSON files, directories, and escaping symlinks to test containment
    outside_json = tmp_path / "outside_secret.json"
    outside_json.write_text('{"secret": true}', encoding="utf-8")
    (bundle_dir / "notes.txt").write_text("internal note", encoding="utf-8")
    (bundle_dir / "scenes" / "pocket.pdb").write_text("ATOM", encoding="utf-8")
    (bundle_dir / "scenes" / "subdir.json").mkdir()
    (bundle_dir / "scenes" / "symlink_escape.json").symlink_to(outside_json)

    # Valid entrypoints
    for entry in ("", "index.html", "interaction_viewer.html", "/index.html"):
        res = resolve_viewer_asset(entry, bundle_dir)
        assert res is not None, f"Expected HTML for {entry!r}"
        body, ctype = res
        assert ctype == "text/html; charset=utf-8"
        assert b"3SYN" in body

    # Unbuilt directory still serves standalone fallback viewer document
    unbuilt_dir = tmp_path / "not_built_yet"
    fallback = resolve_viewer_asset("", unbuilt_dir)
    assert fallback is not None
    assert fallback[1] == "text/html; charset=utf-8"
    assert fallback[0].lower().startswith(b"<!doctype html>")

    # Symlinked interaction_viewer.html pointing outside bundle_dir must be refused
    symlink_html_dir = tmp_path / "symlink_html_bundle"
    symlink_html_dir.mkdir()
    (symlink_html_dir / "interaction_viewer.html").symlink_to(outside_json)
    assert resolve_viewer_asset("", symlink_html_dir) is None
    assert resolve_viewer_asset("interaction_viewer.html", symlink_html_dir) is None

    # Valid scene JSON
    scene_res = resolve_viewer_asset("scenes/3syn.json", bundle_dir)
    assert scene_res is not None
    scene_bytes, scene_ctype = scene_res
    assert scene_ctype == "application/json; charset=utf-8"
    assert json.loads(scene_bytes.decode("utf-8"))["schema_version"] == VIEWER_SCENE_SCHEMA

    # Adversarial traversal payloads & non-JSON / symlink paths -> ALL refused (None)
    forbidden_paths = [
        "../../../etc/passwd",
        "scenes/../../../etc/passwd",
        "../outside_secret.json",
        "scenes/../../outside_secret.json",
        "..%2f..%2f..%2fetc%2fpasswd",
        "scenes/%2e%2e/%2e%2e/outside_secret.json",
        "%2e%2e%2foutside_secret.json",
        "..\\outside_secret.json",
        "scenes/3syn.json\x00.txt",
        "scenes/symlink_escape.json",
        "notes.txt",
        "scenes/pocket.pdb",
        "scenes/subdir.json",
        "scenes/missing.json",
    ]
    for bad in forbidden_paths:
        assert resolve_viewer_asset(bad, bundle_dir) is None, f"Failed to refuse {bad!r}"


def _raw_get(port: int, raw_path: str) -> tuple[int, dict[str, str], bytes]:
    """Send an exact un-normalized HTTP GET request path via http.client."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest("GET", raw_path)
        conn.endheaders()
        resp = conn.getresponse()
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, headers, body
    finally:
        conn.close()


def test_live_http_studio_server_viewer_routes_and_traversal_guard(
    synthetic_complex_files: tuple[Path, Path, Chem.Mol],
    tmp_path: Path,
) -> None:
    """StudioHTTPServer serves /viewer & /viewer/scenes/<id>.json and returns 404 on escapes."""
    pocket_pdb, ligand_sdf, mol = synthetic_complex_files
    bundle_dir = tmp_path / "http_bundle"
    short_traj = {
        "schema_version": "geneuclidean.biophysics.binding_trajectory.v1",
        "frames": [
            {
                "step": 0,
                "time_ps": 0.0,
                "coords": [[0.0, 0.0, 0.0]] * mol.GetNumAtoms(),
                "u_total": -12.5,
            }
        ],
    }
    write_viewer_bundle(
        bundle_dir,
        {"3syn": (pocket_pdb, ligand_sdf)},
        notes={"3syn": "HTTP bridge test"},
        include_esp_samples=False,
        trajectories={"3syn": short_traj},
    )

    outside_json = tmp_path / "secret_outside.json"
    outside_json.write_text('{"leaked": true}', encoding="utf-8")
    (bundle_dir / "scenes" / "leak.json").symlink_to(outside_json)
    (bundle_dir / "README.txt").write_text("do not serve", encoding="utf-8")

    server = create_studio_server(
        "127.0.0.1",
        0,
        cors_origin="http://localhost:5173",
        quiet=True,
        viewer_bundle_dir=bundle_dir,
    )
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # 1. GET /viewer and GET /viewer/ -> 200 HTML + CORS
        for html_route in ("/viewer", "/viewer/", "/viewer/index.html"):
            status, headers, body = _raw_get(port, html_route)
            assert status == 200
            assert headers.get("content-type") == "text/html; charset=utf-8"
            assert headers.get("access-control-allow-origin") == "http://localhost:5173"
            text = body.decode("utf-8")
            assert text.lower().startswith("<!doctype html>")
            assert "3SYN" in text

        # 2. OPTIONS /viewer -> 204 CORS preflight
        req_opt = urllib.request.Request(f"http://127.0.0.1:{port}/viewer", method="OPTIONS")
        with urllib.request.urlopen(req_opt, timeout=10) as resp_opt:
            assert resp_opt.status == 204
            assert resp_opt.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"

        # 3. GET /viewer/scenes/3syn.json and .traj.json -> 200 JSON + CORS
        status_sc, headers_sc, body_sc = _raw_get(port, "/viewer/scenes/3syn.json")
        assert status_sc == 200
        assert headers_sc.get("content-type") == "application/json; charset=utf-8"
        assert headers_sc.get("access-control-allow-origin") == "http://localhost:5173"
        assert json.loads(body_sc.decode("utf-8"))["schema_version"] == VIEWER_SCENE_SCHEMA

        status_tr, headers_tr, body_tr = _raw_get(port, "/viewer/scenes/3syn.traj.json")
        assert status_tr == 200
        assert headers_tr.get("content-type") == "application/json; charset=utf-8"
        assert json.loads(body_tr.decode("utf-8"))["frames"][0]["u_total"] == -12.5

        # 4. Path traversal, URL-encoded traversal, symlink escape, and non-JSON -> 404 JSON
        for attack_route in (
            "/viewer/../../../etc/passwd",
            "/viewer/scenes/../../../etc/passwd",
            "/viewer/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
            "/viewer/..%2fsecret_outside.json",
            "/viewer/scenes/leak.json",
            "/viewer/README.txt",
            "/viewer_nonexistent",
        ):
            st_bad, hdr_bad, body_bad = _raw_get(port, attack_route)
            assert st_bad == 404, f"Expected 404 for {attack_route!r}, got {st_bad}"
            assert hdr_bad.get("content-type") == "application/json; charset=utf-8"
            err_payload: dict[str, Any] = json.loads(body_bad.decode("utf-8"))
            assert err_payload["error"] == "Not found"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # 5. Default create_studio_server (viewer_bundle_dir=None) still serves standalone /viewer
    default_srv = create_studio_server("127.0.0.1", 0, quiet=True)
    default_port = int(default_srv.server_address[1])
    default_thread = threading.Thread(target=default_srv.serve_forever, daemon=True)
    default_thread.start()
    try:
        st_def, hdr_def, body_def = _raw_get(default_port, "/viewer")
        assert st_def == 200
        assert hdr_def.get("content-type") == "text/html; charset=utf-8"
        assert body_def.decode("utf-8").lower().startswith("<!doctype html>")

        st_miss, _, _ = _raw_get(default_port, "/viewer/scenes/3syn.json")
        assert st_miss == 404

        st_trav, _, _ = _raw_get(default_port, "/viewer/../server.py")
        assert st_trav == 404
    finally:
        default_srv.shutdown()
        default_srv.server_close()
        default_thread.join(timeout=5)
