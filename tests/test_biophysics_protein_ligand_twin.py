"""Google-SWE Verification Suite for ``geneuclidean.biophysics`` 3D Digital Twin.

Validates:
1. ``< 15 ms`` cold-import latency of ``geneuclidean.biophysics`` with zero eager imports
   of ``torch``, ``scipy``, ``sklearn``, ``rdkit``, ``matplotlib``, ``Bio``, ``transformers``,
   or ``peft``, plus sorted ``__all__`` and AST security compliance.
2. Strict ``torch.float32`` discipline (zero ``torch.float64`` parameters, buffers, or
   outputs) for Apple Silicon MPS / CUDA / CPU portability.
3. Electronegativity equalization / Gasteiger–Marsili formal-charge conservation on neutral,
   cationic, and anionic ligands, plus AMBER-inspired protein residue formal charges.
4. Exact ``SE(3)`` rotation + translation invariance of total potential energy
   :math:`U(R\\mathbf{x}+\\mathbf{t}) == U(\\mathbf{x})`, exact ``SO(3)`` equivariance of
   force vectors :math:`\\vec{F}(R\\mathbf{x}+\\mathbf{t}) == R\\vec{F}(\\mathbf{x})` and
   3D electric fields :math:`\\vec{E}(R\\mathbf{x}+\\mathbf{t}) == R\\vec{E}(\\mathbf{x})`,
   and translational momentum conservation (:math:`\\sum \\vec{F} = \\vec{0}`).
5. Central finite-difference gradient verification matching exact PyTorch autograd forces
   :math:`-\\nabla_{\\mathbf{r}} U_{\\text{total}}` across all physical & electrical terms.
6. Salt-bridge attraction (:math:`-\\Delta E`) vs. like-charge repulsion (:math:`+\\Delta E`),
   OBC2 Generalized Born polar desolvation penalty in hydrophobic cavities, soft-core
   :math:`r_{ij}=0` clash finiteness, linear :math:`180^\\circ` :math:`\\text{C–Br}\\cdots\\text{O}`
   :math:`\\sigma`-hole halogen bond minimum vs. :math:`90^\\circ` lateral belt repulsion,
   and cation–:math:`\\pi` / :math:`\\pi`–:math:`\\pi` quadrupole orientation dependence.
7. ``ProteinLigandDigitalTwin`` end-to-end ``relax_complex`` clash relief & stabilization,
   ``sample_thermal_ensemble`` 300 K Langevin RMSF & contact occupancy statistics,
   ``guidance_force_fn`` batched/unbatched 3D flow guidance, ``OpenMMTwinAdapter``,
   ``compute_electrostatic_potential_grid``, ``extract_typed_interaction_network``,
   ``render_biophysical_twin_figure``, ``export_web_twin_scene_json``, and
   ``export_interactive_twin_html``.
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

import geneuclidean.biophysics as bp
from geneuclidean.biophysics import (
    BiophysicalAtomParams,
    BiophysicalHamiltonian,
    OpenMMTwinAdapter,
    ProteinLigandDigitalTwin,
    compute_electrostatic_potential_grid,
    export_interactive_twin_html,
    export_web_twin_scene_json,
    extract_typed_interaction_network,
    parameterize_ligand_biophysics,
    parameterize_pocket_biophysics,
    render_biophysical_twin_figure,
)
from geneuclidean.exceptions import (
    InvalidMoleculeError,
    OptionalDependencyError,
    StructureNotFoundError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BIOPHYS_DIR = REPO_ROOT / "src" / "geneuclidean" / "biophysics"

HEAVY_MODULES: tuple[str, ...] = (
    "torch",
    "numpy",
    "transformers",
    "peft",
    "sklearn",
    "scipy",
    "rdkit",
    "matplotlib",
    "Bio",
)


# ---------------------------------------------------------------------------
# Realistic Synthetic CNS GPCR-Style Binding Pocket PDB Fixture
# ---------------------------------------------------------------------------

SYNTHETIC_GPCR_POCKET_PDB = """\
HEADER    AMINERGIC GPCR ORTHOSTERIC BINDING POCKET
ATOM      1  N   ASP A 103      -7.200   1.900  -1.200  1.00 20.00           N
ATOM      2  CA  ASP A 103      -6.600   0.600  -0.900  1.00 20.00           C
ATOM      3  C   ASP A 103      -7.350  -0.550  -1.550  1.00 20.00           C
ATOM      4  O   ASP A 103      -8.550  -0.650  -1.450  1.00 20.00           O
ATOM      5  CB  ASP A 103      -6.500   0.350   0.610  1.00 20.00           C
ATOM      6  CG  ASP A 103      -5.850  -0.980   0.950  1.00 20.00           C
ATOM      7  OD1 ASP A 103      -5.300  -1.750   0.100  1.00 20.00           O
ATOM      8  OD2 ASP A 103      -5.500  -1.200   2.150  1.00 20.00           O
ATOM      9  N   SER A 193       7.850   2.600  -0.800  1.00 20.00           N
ATOM     10  CA  SER A 193       7.350   1.250  -0.550  1.00 20.00           C
ATOM     11  C   SER A 193       8.250   0.200  -1.180  1.00 20.00           C
ATOM     12  O   SER A 193       7.800  -0.920  -1.420  1.00 20.00           O
ATOM     13  CB  SER A 193       7.200   0.980   0.950  1.00 20.00           C
ATOM     14  OG  SER A 193       6.100   1.800   1.450  1.00 20.00           O
ATOM     15  N   PHE A 389       0.200  -6.200  -2.200  1.00 20.00           N
ATOM     16  CA  PHE A 389       0.150  -5.800  -0.780  1.00 20.00           C
ATOM     17  C   PHE A 389       1.450  -6.150  -0.080  1.00 20.00           C
ATOM     18  O   PHE A 389       2.520  -5.800  -0.560  1.00 20.00           O
ATOM     19  CB  PHE A 389      -1.050  -6.420  -0.060  1.00 20.00           C
ATOM     20  CG  PHE A 389      -1.020  -5.900   1.380  1.00 20.00           C
ATOM     21  CD1 PHE A 389      -0.020  -5.050   1.900  1.00 20.00           C
ATOM     22  CE1 PHE A 389       0.010  -4.650   3.240  1.00 20.00           C
ATOM     23  CZ  PHE A 389      -0.960  -5.100   4.120  1.00 20.00           C
ATOM     24  CE2 PHE A 389      -1.960  -5.950   3.600  1.00 20.00           C
ATOM     25  CD2 PHE A 389      -1.990  -6.350   2.260  1.00 20.00           C
ATOM     26  N   LYS A 190       0.800   6.900   0.200  1.00 20.00           N
ATOM     27  CA  LYS A 190       0.550   6.300   1.520  1.00 20.00           C
ATOM     28  C   LYS A 190       1.820   6.150   2.350  1.00 20.00           C
ATOM     29  O   LYS A 190       2.910   6.450   1.870  1.00 20.00           O
ATOM     30  CB  LYS A 190      -0.480   7.100   2.320  1.00 20.00           C
ATOM     31  CG  LYS A 190      -0.850   6.450   3.650  1.00 20.00           C
ATOM     32  CD  LYS A 190      -0.100   5.150   3.920  1.00 20.00           C
ATOM     33  CE  LYS A 190      -0.420   4.550   5.280  1.00 20.00           C
ATOM     34  NZ  LYS A 190       0.250   3.300   5.480  1.00 20.00           N
END
"""


def _random_so3_matrix(seed: int = 17) -> torch.Tensor:
    """Construct a deterministic proper rotation matrix ``R in SO(3)`` (``det(R) = +1``)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(raw)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] = -q[:, 0]
    return torch.from_numpy(q.astype(np.float32))


def _make_single_atom_params(
    xyz: tuple[float, float, float],
    *,
    z: int,
    charge: float,
    sigma: float = 3.25,
    epsilon: float = 0.17,
    born_r: float = 1.55,
    is_donor: bool = False,
    is_acceptor: bool = False,
    is_cationic: bool = False,
    is_anionic: bool = False,
    is_hydrophobic: bool = False,
    res_label: str = "RES1:A",
) -> BiophysicalAtomParams:
    """Create a minimal 1-atom ``BiophysicalAtomParams`` bundle for exact physics probes."""
    return BiophysicalAtomParams(
        coords=torch.tensor([list(xyz)], dtype=torch.float32),
        atomic_nums=torch.tensor([z], dtype=torch.long),
        partial_charges=torch.tensor([charge], dtype=torch.float32),
        vdw_sigma=torch.tensor([sigma], dtype=torch.float32),
        vdw_epsilon=torch.tensor([epsilon], dtype=torch.float32),
        born_radii=torch.tensor([born_r], dtype=torch.float32),
        obc_scaling=torch.tensor([0.80], dtype=torch.float32),
        polarizability=torch.tensor([1.00], dtype=torch.float32),
        is_donor=torch.tensor([is_donor], dtype=torch.bool),
        is_acceptor=torch.tensor([is_acceptor], dtype=torch.bool),
        is_cationic=torch.tensor([is_cationic], dtype=torch.bool),
        is_anionic=torch.tensor([is_anionic], dtype=torch.bool),
        is_hydrophobic=torch.tensor([is_hydrophobic], dtype=torch.bool),
        is_backbone=torch.tensor([False], dtype=torch.bool),
        is_halogen=torch.tensor([False], dtype=torch.bool),
        halogen_strength=torch.tensor([0.0], dtype=torch.float32),
        donor_axes=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        halogen_sigma_axes=torch.zeros((1, 3), dtype=torch.float32),
        ring_centroids=torch.zeros((0, 3), dtype=torch.float32),
        ring_normals=torch.zeros((0, 3), dtype=torch.float32),
        ring_atom_indices=torch.zeros((0, 6), dtype=torch.long),
        ring_atom_mask=torch.zeros((0, 6), dtype=torch.bool),
        halogen_bond_pairs=torch.zeros((0, 2), dtype=torch.long),
        donor_h_pairs=torch.zeros((0, 2), dtype=torch.long),
        bond_pairs=torch.zeros((0, 2), dtype=torch.long),
        bond_eq_lengths=torch.zeros((0,), dtype=torch.float32),
        angle_triplets=torch.zeros((0, 3), dtype=torch.long),
        angle_eq_cos=torch.zeros((0,), dtype=torch.float32),
        residue_indices=torch.tensor([0], dtype=torch.long),
        residue_labels=[res_label],
        atom_names=["A1"],
        elements=["N" if z == 7 else ("O" if z == 8 else "C")],
    )


def _build_demo_twin() -> ProteinLigandDigitalTwin:
    """Build a realistic brominated aminergic ligand + GPCR pocket digital twin."""
    # Protonated 4-bromophenethylammonium: carries +1 NH3+, aromatic ring, and C-Br sigma-hole
    smiles = "[NH3+]CCc1ccc(Br)cc1"
    twin = ProteinLigandDigitalTwin.from_pocket_and_smiles(
        SYNTHETIC_GPCR_POCKET_PDB,
        smiles,
        center=(0.6, -0.3, 1.8),
        seed=42,
    )
    return twin


# ---------------------------------------------------------------------------
# 1. Cold-Import Latency (< 15 ms), Lazy __all__, AST Security & Float32 Hygiene
# ---------------------------------------------------------------------------


def test_biophysics_cold_import_under_15ms_and_zero_heavy_modules() -> None:
    """``import geneuclidean.biophysics`` must finish in < 15 ms with zero heavy ML imports."""
    probe = f"""
import json, sys, time
t0 = time.perf_counter()
import geneuclidean.biophysics
elapsed_ms = (time.perf_counter() - t0) * 1000.0
heavy = [m for m in {HEAVY_MODULES!r} if m in sys.modules]
print(json.dumps({{"elapsed_ms": round(elapsed_ms, 3), "heavy": heavy}}))
"""
    best_ms = float("inf")
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout.strip())
        assert payload["heavy"] == [], f"Leaked eager heavy imports: {payload['heavy']}"
        best_ms = min(best_ms, float(payload["elapsed_ms"]))
    assert best_ms < 15.0, f"Cold import took {best_ms:.2f} ms (>= 15.0 ms budget)"


def test_biophysics_lazy_facade_all_sorted_and_complete() -> None:
    """All symbols in ``geneuclidean.biophysics.__all__`` must be sorted and resolve cleanly."""
    mod = importlib.import_module("geneuclidean.biophysics")
    names = list(mod.__all__)
    assert len(names) >= 16
    assert names == sorted(names)
    assert len(names) == len(set(names))
    dir_entries = set(dir(mod))
    for sym in names:
        assert sym in dir_entries
        assert getattr(mod, sym) is not None

    with pytest.raises(AttributeError):
        _ = mod.__nonexistent_biophys_symbol__


def test_biophysics_ast_security_zero_eval_exec_shell_or_os_system() -> None:
    """AST scan of ``src/geneuclidean/biophysics/*.py`` forbids eval/exec/shell=True/os.system."""
    py_files = sorted(BIOPHYS_DIR.glob("*.py"))
    # The point of this check is that the glob actually found the modules it claims to
    # scan -- an empty glob would make the whole test pass vacuously. Pinning an exact
    # count also made it fail every time a module was added, which says nothing about
    # security, so assert the known modules are covered instead.
    found = {path.name for path in py_files}
    expected = {"__init__.py", "forcefield.py", "interaction_viz.py", "protein_ligand_twin.py"}
    assert expected <= found, f"AST scan is missing {sorted(expected - found)}"
    violations: list[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                violations.append(f"{path.name}:{node.lineno}:{node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                violations.append(f"{path.name}:{node.lineno}:os.system")
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    violations.append(f"{path.name}:{node.lineno}:shell=True")
    assert violations == []


def test_zero_float64_params_buffers_and_outputs() -> None:
    """Verify strictly zero ``torch.float64`` state across Hamiltonian, params, and outputs."""
    twin = _build_demo_twin()
    ham = twin.hamiltonian

    for name, p in ham.named_parameters():
        assert p.dtype != torch.float64, f"float64 param: {name}"
    for name, b in ham.named_buffers():
        assert b.dtype != torch.float64, f"float64 buffer: {name}"

    for label, params in (("ligand", twin.ligand_params), ("pocket", twin.pocket_params)):
        for attr_name, val in vars(params).items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert val.dtype == torch.float32, f"{label}.{attr_name} is {val.dtype}"

    bd = twin.evaluate_complex()
    for field_name in (
        "ligand_forces_total",
        "pocket_forces_total",
        "ligand_forces_elec",
        "pocket_forces_elec",
        "ligand_forces_vdw",
        "pocket_forces_vdw",
        "ligand_electric_field",
        "pocket_electric_field",
        "ligand_born_radii",
        "pocket_born_radii",
    ):
        t_val = getattr(bd, field_name)
        assert t_val.dtype == torch.float32, f"Breakdown {field_name} has {t_val.dtype}"


# ---------------------------------------------------------------------------
# 2. Biophysical Parameterization & Formal-Charge Conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected_formal_q"),
    [
        ("c1ccc(Br)cc1", 0.0),
        ("[NH3+]CCc1ccc(Cl)cc1", 1.0),
        ("CC(=O)[O-]", -1.0),
    ],
)
def test_ligand_parameterization_conserves_formal_charge_and_extracts_features(
    smiles: str,
    expected_formal_q: float,
) -> None:
    """Ligand partial charges must sum strictly to net formal charge."""
    params = parameterize_ligand_biophysics(smiles)
    q_sum = float(params.partial_charges.sum().item())
    assert math.isclose(q_sum, expected_formal_q, abs_tol=1e-4)
    assert math.isclose(params.net_formal_charge, expected_formal_q, abs_tol=1e-5)

    if "Br" in smiles or "Cl" in smiles:
        assert bool(params.is_halogen.any())
        assert params.halogen_bond_pairs.shape[0] == 1
        hal_norms = torch.linalg.vector_norm(params.halogen_sigma_axes[params.is_halogen], dim=-1)
        assert torch.allclose(hal_norms, torch.ones_like(hal_norms), atol=1e-4)

    if "c1" in smiles:
        assert params.ring_centroids.shape == (1, 3)
        assert params.ring_normals.shape == (1, 3)

    with pytest.raises(InvalidMoleculeError):
        parameterize_ligand_biophysics("NOT_A_VALID_SMILES_%%%")


def test_pocket_parameterization_charges_ionizable_residues_accurately() -> None:
    """ASP103 must carry net -1 charge, LYS190 net +1, and neutral SER193/PHE389 net 0."""
    rec = parameterize_pocket_biophysics(SYNTHETIC_GPCR_POCKET_PDB)
    res_idx = rec.residue_indices.numpy()
    q_np = rec.partial_charges.numpy()

    charge_by_res = {
        label: float(q_np[res_idx == idx].sum()) for idx, label in enumerate(rec.residue_labels)
    }
    assert math.isclose(charge_by_res["ASP103:A"], -1.0, abs_tol=1e-3)
    assert math.isclose(charge_by_res["LYS190:A"], 1.0, abs_tol=1e-3)
    assert math.isclose(charge_by_res["SER193:A"], 0.0, abs_tol=1e-3)
    assert math.isclose(charge_by_res["PHE389:A"], 0.0, abs_tol=1e-3)
    assert rec.ring_centroids.shape == (1, 3)  # PHE389 benzene ring

    with pytest.raises(StructureNotFoundError):
        parameterize_pocket_biophysics("/nonexistent/path/to/missing_pocket.pdb")


# ---------------------------------------------------------------------------
# 3. Exact SE(3) Invariance, SO(3) Force/Field Equivariance & Finite Differences
# ---------------------------------------------------------------------------


@pytest.mark.equivariance
def test_se3_energy_invariance_and_so3_force_efield_equivariance() -> None:
    """Verify U(Rx + t) == U(x), F(Rx + t) == R F(x), E(Rx + t) == R E(x), and sum(F) == 0."""
    twin = _build_demo_twin()
    twin.relax_complex(steps=25, flexible_sidechains=False)
    ham = twin.hamiltonian
    lig = twin.ligand_params
    rec = twin.pocket_params

    bd_ref = ham.evaluate_complex(lig, rec)

    # Newton's third law / translational invariance: net complex force vanishes
    net_force = bd_ref.ligand_forces_total.sum(dim=0) + bd_ref.pocket_forces_total.sum(dim=0)
    assert torch.allclose(net_force, torch.zeros(3, dtype=torch.float32), atol=1e-3)

    rot = _random_so3_matrix(seed=99)
    trans = torch.tensor([37.5, -24.25, 18.75], dtype=torch.float32)

    lc_rot = lig.coords @ rot.T + trans
    rc_rot = rec.coords @ rot.T + trans

    bd_rot = ham.evaluate_complex(lig, rec, lig_coords=lc_rot, rec_coords=rc_rot)

    # SE(3) scalar energy invariance across total and individual terms
    assert math.isclose(bd_rot.u_total, bd_ref.u_total, abs_tol=2e-4)
    assert math.isclose(bd_rot.e_coulomb, bd_ref.e_coulomb, abs_tol=1e-4)
    assert math.isclose(bd_rot.dg_gb_solvation, bd_ref.dg_gb_solvation, abs_tol=1e-4)
    assert math.isclose(bd_rot.e_vdw, bd_ref.e_vdw, abs_tol=1e-4)
    assert math.isclose(bd_rot.e_hbond, bd_ref.e_hbond, abs_tol=1e-4)
    assert math.isclose(bd_rot.e_halogen, bd_ref.e_halogen, abs_tol=1e-4)
    assert math.isclose(bd_rot.e_cation_pi, bd_ref.e_cation_pi, abs_tol=1e-4)
    assert math.isclose(bd_rot.e_pi_stacking, bd_ref.e_pi_stacking, abs_tol=1e-4)
    assert math.isclose(bd_rot.dg_hydrophobic, bd_ref.dg_hydrophobic, abs_tol=1e-4)

    # SO(3) vector equivariance of total forces, electrical forces, vdW forces & E-field
    assert torch.allclose(bd_rot.ligand_forces_total, bd_ref.ligand_forces_total @ rot.T, atol=2e-3)
    assert torch.allclose(bd_rot.pocket_forces_total, bd_ref.pocket_forces_total @ rot.T, atol=2e-3)
    assert torch.allclose(bd_rot.ligand_forces_elec, bd_ref.ligand_forces_elec @ rot.T, atol=2e-3)
    assert torch.allclose(bd_rot.ligand_forces_vdw, bd_ref.ligand_forces_vdw @ rot.T, atol=2e-3)
    assert torch.allclose(
        bd_rot.ligand_electric_field, bd_ref.ligand_electric_field @ rot.T, atol=1e-3
    )
    assert torch.allclose(
        bd_rot.pocket_electric_field, bd_ref.pocket_electric_field @ rot.T, atol=1e-3
    )


def test_analytical_autograd_forces_match_central_finite_differences() -> None:
    """Verify exact analytical forces ``F_i = -grad U`` match central finite differences."""
    twin = _build_demo_twin()
    # Relax slightly first so forces are moderate (~1-15 kcal/(mol*A)) for float32 FD checks
    twin.relax_complex(steps=30, flexible_sidechains=False)

    ham = twin.hamiltonian
    lig = twin.ligand_params
    rec = twin.pocket_params
    bd = ham.evaluate_complex(lig, rec)

    h = 2.0e-3  # Optimal step size for float32 central finite differences
    lc_base = lig.coords.detach().clone()
    rc_base = rec.coords.detach().clone()

    # Probe first 8 ligand atoms across all 3 Cartesian axes
    n_probe_l = min(8, lig.num_atoms)
    fd_lig = torch.zeros((n_probe_l, 3), dtype=torch.float32)
    with torch.no_grad():
        for i in range(n_probe_l):
            for d in range(3):
                lc_p = lc_base.clone()
                lc_m = lc_base.clone()
                lc_p[i, d] += h
                lc_m[i, d] -= h
                u_p = float(ham.potential_energy(lc_p, rc_base, lig, rec).item())
                u_m = float(ham.potential_energy(lc_m, rc_base, lig, rec).item())
                fd_lig[i, d] = -(u_p - u_m) / (2.0 * h)

    err_lig = torch.linalg.vector_norm(bd.ligand_forces_total[:n_probe_l] - fd_lig, dim=-1)
    assert float(err_lig.max().item()) < 5e-2, f"Max ligand FD force error: {err_lig.max().item()}"

    # Probe 6 pocket atoms near the binding site across all 3 Cartesian axes
    fd_rec = torch.zeros((6, 3), dtype=torch.float32)
    probe_rec_indices = [5, 6, 7, 13, 20, 33]  # ASP OD1/OD2, SER OG, PHE CG, LYS NZ
    with torch.no_grad():
        for k_pos, rj in enumerate(probe_rec_indices):
            for d in range(3):
                rc_p = rc_base.clone()
                rc_m = rc_base.clone()
                rc_p[rj, d] += h
                rc_m[rj, d] -= h
                u_p = float(ham.potential_energy(lc_base, rc_p, lig, rec).item())
                u_m = float(ham.potential_energy(lc_base, rc_m, lig, rec).item())
                fd_rec[k_pos, d] = -(u_p - u_m) / (2.0 * h)

    analytic_rec = bd.pocket_forces_total[probe_rec_indices]
    err_rec = torch.linalg.vector_norm(analytic_rec - fd_rec, dim=-1)
    assert float(err_rec.max().item()) < 5e-2, f"Max pocket FD force error: {err_rec.max().item()}"


# ---------------------------------------------------------------------------
# 4. Physical Chemistry Laws: Electrostatics, GB Desolvation, Sigma-Hole & Pi
# ---------------------------------------------------------------------------


def test_salt_bridge_attraction_vs_like_charge_repulsion_and_gb_desolvation() -> None:
    """Opposite charges attract (-dE), like charges repel (+dE), and burial costs +dG_GB."""
    ham = BiophysicalHamiltonian()

    lig_pos = _make_single_atom_params((0.0, 0.0, 0.0), z=7, charge=1.0, is_cationic=True)
    rec_neg = _make_single_atom_params(
        (3.1, 0.0, 0.0), z=8, charge=-1.0, is_anionic=True, res_label="ASP1:A"
    )
    rec_pos = _make_single_atom_params(
        (3.1, 0.0, 0.0), z=7, charge=1.0, is_cationic=True, res_label="LYS2:A"
    )

    bd_attract = ham.evaluate_complex(lig_pos, rec_neg)
    bd_repel = ham.evaluate_complex(lig_pos, rec_pos)

    assert bd_attract.e_coulomb < -6.0
    assert (bd_attract.e_coulomb + bd_attract.dg_gb_solvation) < -5.0
    assert bd_attract.u_total < -5.0
    # Attractive force pulls ligand (+x direction toward x=3.1)
    assert float(bd_attract.ligand_forces_total[0, 0].item()) > 1.0

    assert bd_repel.e_coulomb > 6.0
    assert (bd_repel.e_coulomb + bd_repel.dg_gb_solvation) > 6.0
    assert bd_repel.u_total > 6.0
    # Repulsive force pushes ligand away (-x direction)
    assert float(bd_repel.ligand_forces_total[0, 0].item()) < -1.0

    # OBC2 desolvation: burying +1 ion in neutral hydrophobic shell raises Born radius
    shell_angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
    shell_xyz = np.stack(
        [3.6 * np.cos(shell_angles), 3.6 * np.sin(shell_angles), np.zeros_like(shell_angles)],
        axis=-1,
    ).astype(np.float32)
    n_s = len(shell_xyz)
    hydrophobic_shell = BiophysicalAtomParams(
        coords=torch.from_numpy(shell_xyz),
        atomic_nums=torch.full((n_s,), 6, dtype=torch.long),
        partial_charges=torch.zeros((n_s,), dtype=torch.float32),
        vdw_sigma=torch.full((n_s,), 3.399, dtype=torch.float32),
        vdw_epsilon=torch.full((n_s,), 0.086, dtype=torch.float32),
        born_radii=torch.full((n_s,), 1.70, dtype=torch.float32),
        obc_scaling=torch.full((n_s,), 0.72, dtype=torch.float32),
        polarizability=torch.full((n_s,), 1.20, dtype=torch.float32),
        is_donor=torch.zeros((n_s,), dtype=torch.bool),
        is_acceptor=torch.zeros((n_s,), dtype=torch.bool),
        is_cationic=torch.zeros((n_s,), dtype=torch.bool),
        is_anionic=torch.zeros((n_s,), dtype=torch.bool),
        is_hydrophobic=torch.ones((n_s,), dtype=torch.bool),
        is_backbone=torch.zeros((n_s,), dtype=torch.bool),
        is_halogen=torch.zeros((n_s,), dtype=torch.bool),
        halogen_strength=torch.zeros((n_s,), dtype=torch.float32),
        donor_axes=torch.zeros((n_s, 3), dtype=torch.float32),
        halogen_sigma_axes=torch.zeros((n_s, 3), dtype=torch.float32),
        ring_centroids=torch.zeros((0, 3), dtype=torch.float32),
        ring_normals=torch.zeros((0, 3), dtype=torch.float32),
        ring_atom_indices=torch.zeros((0, 6), dtype=torch.long),
        ring_atom_mask=torch.zeros((0, 6), dtype=torch.bool),
        halogen_bond_pairs=torch.zeros((0, 2), dtype=torch.long),
        donor_h_pairs=torch.zeros((0, 2), dtype=torch.long),
        bond_pairs=torch.zeros((0, 2), dtype=torch.long),
        bond_eq_lengths=torch.zeros((0,), dtype=torch.float32),
        angle_triplets=torch.zeros((0, 3), dtype=torch.long),
        angle_eq_cos=torch.zeros((0,), dtype=torch.float32),
        residue_indices=torch.zeros((n_s,), dtype=torch.long),
        residue_labels=["LEU10:A"],
        atom_names=[f"C{i}" for i in range(n_s)],
        elements=["C"] * n_s,
    )
    bd_buried = ham.evaluate_complex(lig_pos, hydrophobic_shell)
    # Burial must raise the effective Born radius relative to the *unburied* effective
    # radius. Comparing against the intrinsic radius is the wrong baseline: OBC2 subtracts
    # a 0.09 A dielectric offset, so an atom with no descreening neighbour sits at
    # rho - 0.09 (1.46 A here), below its intrinsic 1.55 A. The previous assertion compared
    # against 1.55 and passed only because a saturating burial integral pinned every atom
    # at the 12 A ceiling -- which also forced dg_gb_solvation to exactly 0.0 on every real
    # complex. See BiophysicalHamiltonian.compute_effective_born_radii.
    r_free = float(
        ham.compute_effective_born_radii(lig_pos.coords, lig_pos.born_radii, lig_pos.obc_scaling)[
            0
        ].item()
    )
    r_bound = float(bd_buried.ligand_born_radii[0].item())
    assert r_bound > r_free, f"burial must raise R_eff: free {r_free:.3f} -> bound {r_bound:.3f}"
    assert r_bound < 12.0, "a shallow 12-atom shell must not saturate the radius ceiling"
    assert bd_buried.dg_gb_solvation > 0.0  # Positive polar desolvation penalty

    # Analytical HCT engulfment branch check: when rho_i < sr_j - d, the 0.5 prefactor
    # must scale BOTH `bracket` and `2 * (1/rho_i - 1/L)` (OpenMM GBSAOBCForce convention).
    engulf_coords = torch.tensor([[0.0, 0.0, 0.0], [0.40, 0.0, 0.0]], dtype=torch.float32)
    engulf_rho = torch.tensor([1.20, 2.09], dtype=torch.float32)  # rho_t = [1.11, 2.00]
    engulf_obc = torch.tensor([0.85, 0.95], dtype=torch.float32)  # sr_1 = 1.90 > 1.11 + 0.40
    r_engulf = float(
        ham.compute_effective_born_radii(engulf_coords, engulf_rho, engulf_obc)[0].item()
    )
    d_pair, rho0, rho_t0, sr1 = 0.40, 1.20, 1.11, 1.90
    u_lim, l_lim = d_pair + sr1, max(rho_t0, abs(d_pair - sr1))
    bracket = (
        (1.0 / l_lim - 1.0 / u_lim)
        + 0.25 * (d_pair - sr1 * sr1 / d_pair) * (1.0 / (u_lim * u_lim) - 1.0 / (l_lim * l_lim))
        + 0.5 * math.log(l_lim / u_lim) / d_pair
    )
    i_exact = 0.5 * (bracket + 2.0 * (1.0 / rho_t0 - 1.0 / l_lim))
    psi_ex = i_exact * rho_t0
    r_expected = 1.0 / (
        (1.0 / rho_t0) - math.tanh(psi_ex * (1.0 - 0.8 * psi_ex + 4.85 * psi_ex**2)) / rho0
    )
    assert math.isclose(r_engulf, r_expected, rel_tol=1e-4)


def test_linear_180_deg_c_br_o_sigma_hole_minimum_vs_90_deg_lateral_clash() -> None:
    """Linear 180° C-Br...O sigma-hole must be strongly favored over 90° lateral approach."""
    ham = BiophysicalHamiltonian()

    # Construct explicit 2-atom C--Br fragment along +x: C at (-1.90, 0, 0), Br at (0, 0, 0)
    c_br_lig = BiophysicalAtomParams(
        coords=torch.tensor([[-1.90, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
        atomic_nums=torch.tensor([6, 35], dtype=torch.long),
        partial_charges=torch.tensor([0.08, -0.08], dtype=torch.float32),
        vdw_sigma=torch.tensor([3.399, 3.596], dtype=torch.float32),
        vdw_epsilon=torch.tensor([0.086, 0.320], dtype=torch.float32),
        born_radii=torch.tensor([1.70, 1.85], dtype=torch.float32),
        obc_scaling=torch.tensor([0.72, 0.80], dtype=torch.float32),
        polarizability=torch.tensor([1.35, 3.01], dtype=torch.float32),
        is_donor=torch.tensor([False, False], dtype=torch.bool),
        is_acceptor=torch.tensor([False, False], dtype=torch.bool),
        is_cationic=torch.tensor([False, False], dtype=torch.bool),
        is_anionic=torch.tensor([False, False], dtype=torch.bool),
        is_hydrophobic=torch.tensor([False, False], dtype=torch.bool),
        is_backbone=torch.tensor([False, False], dtype=torch.bool),
        is_halogen=torch.tensor([False, True], dtype=torch.bool),
        halogen_strength=torch.tensor([0.0, 1.85], dtype=torch.float32),
        donor_axes=torch.zeros((2, 3), dtype=torch.float32),
        halogen_sigma_axes=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        ring_centroids=torch.zeros((0, 3), dtype=torch.float32),
        ring_normals=torch.zeros((0, 3), dtype=torch.float32),
        ring_atom_indices=torch.zeros((0, 6), dtype=torch.long),
        ring_atom_mask=torch.zeros((0, 6), dtype=torch.bool),
        halogen_bond_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        donor_h_pairs=torch.zeros((0, 2), dtype=torch.long),
        bond_pairs=torch.tensor([[0, 1]], dtype=torch.long),
        bond_eq_lengths=torch.tensor([1.90], dtype=torch.float32),
        angle_triplets=torch.zeros((0, 3), dtype=torch.long),
        angle_eq_cos=torch.zeros((0,), dtype=torch.float32),
        residue_indices=torch.zeros((2,), dtype=torch.long),
        residue_labels=["LIG1:L"],
        atom_names=["C1", "BR1"],
        elements=["C", "Br"],
    )

    r_xb = 3.30
    # 180° linear approach: carbonyl acceptor O at (+3.30, 0, 0) => angle C-Br...O == 180°
    o_linear = _make_single_atom_params(
        (r_xb, 0.0, 0.0), z=8, charge=-0.55, is_acceptor=True, res_label="SER193:A"
    )
    # 90° lateral approach at identical Br...O distance: O at (0, +3.30, 0) => angle C-Br...O == 90°
    o_lateral = _make_single_atom_params(
        (0.0, r_xb, 0.0), z=8, charge=-0.55, is_acceptor=True, res_label="SER193:A"
    )

    bd_180 = ham.evaluate_complex(c_br_lig, o_linear)
    bd_90 = ham.evaluate_complex(c_br_lig, o_lateral)

    assert bd_180.e_halogen < -3.0, f"Expected strong 180° sigma-hole well, got {bd_180.e_halogen}"
    assert bd_90.e_halogen > 1.0, f"Expected 90° equatorial belt penalty, got {bd_90.e_halogen}"
    assert bd_180.u_total < bd_90.u_total - 4.0

    # Verify extracted interaction network detects halogen_sigma_hole at 180° and not at 90°
    twin_180 = ProteinLigandDigitalTwin.from_tensors(c_br_lig, o_linear, hamiltonian=ham)
    twin_90 = ProteinLigandDigitalTwin.from_tensors(c_br_lig, o_lateral, hamiltonian=ham)
    xb_180 = [
        x for x in extract_typed_interaction_network(twin_180) if x.kind == "halogen_sigma_hole"
    ]
    xb_90 = [
        x for x in extract_typed_interaction_network(twin_90) if x.kind == "halogen_sigma_hole"
    ]
    assert len(xb_180) == 1
    assert xb_180[0].angle_deg is not None and math.isclose(xb_180[0].angle_deg, 180.0, abs_tol=1.0)
    assert xb_90 == []


def test_softcore_coincident_atom_clash_is_finite_without_nan_or_inf() -> None:
    """Coincident r_ij = 0.0 Å coordinates under soft-core lambda < 1 must yield finite forces."""
    ham = BiophysicalHamiltonian()
    lig_a = _make_single_atom_params((1.0, 2.0, 3.0), z=6, charge=0.25)
    rec_a = _make_single_atom_params((1.0, 2.0, 3.0), z=6, charge=-0.25)

    bd_sc = ham.evaluate_complex(lig_a, rec_a, softcore_lambda=0.5)
    assert math.isfinite(bd_sc.u_total)
    assert bool(torch.isfinite(bd_sc.ligand_forces_total).all())
    assert bool(torch.isfinite(bd_sc.pocket_forces_total).all())


# ---------------------------------------------------------------------------
# 5. Induced-Fit Relaxation, Thermal Langevin Ensemble, Flow Guidance & OpenMM
# ---------------------------------------------------------------------------


def test_relax_complex_resolves_planted_steric_clash_and_lowers_energy() -> None:
    """Planting a severe steric overlap must be resolved cleanly by ``relax_complex``."""
    twin = _build_demo_twin()

    # Plant a direct heavy-atom clash near ASP103 OD1 (-2.20, -1.80, 0.04)
    asp_od1_xyz = twin.pocket_params.coords[6].clone()
    lig_heavy_idx = int(torch.nonzero(twin.ligand_params.atomic_nums > 1, as_tuple=False)[0].item())
    shift_into_clash = (
        asp_od1_xyz + torch.tensor([1.05, 0.15, 0.10], dtype=torch.float32)
    ) - twin.ligand_params.coords[lig_heavy_idx]
    clashed_coords = twin.ligand_params.coords + shift_into_clash.unsqueeze(0)
    twin.ligand_params = twin.ligand_params.with_coords(clashed_coords)

    pre_inspect = twin.inspect_physics()
    assert pre_inspect.clash_diagnostics["pauli_clash_count"] >= 1
    assert pre_inspect.u_total > 20.0

    res = twin.relax_complex(steps=55, lr=0.05, optimizer="lbfgs", flexible_sidechains=True)
    assert res.initial_clash_count >= 1
    assert res.final_clash_count == 0
    assert res.final_energy < res.initial_energy - 20.0
    assert res.delta_energy < -20.0
    assert res.final_min_dist_angstrom > res.initial_min_dist_angstrom
    assert res.backbone_rmsd_angstrom == 0.0  # Backbone strictly anchored
    assert res.converged is True

    post_inspect = twin.inspect_physics()
    assert post_inspect.clash_diagnostics["clash_free"] is True
    assert len(post_inspect.hotspot_table) == len(twin.pocket_params.residue_labels)


def test_sample_thermal_ensemble_returns_finite_rmsf_and_occupancy_stats() -> None:
    """300 K overdamped Langevin sampling returns finite RMSFs, occupancies, and preserves pose."""
    twin = _build_demo_twin()
    twin.relax_complex(steps=35, flexible_sidechains=True)
    eq_lig_coords = twin.ligand_params.coords.clone()
    eq_rec_coords = twin.pocket_params.coords.clone()

    summary = twin.sample_thermal_ensemble(
        n_steps=32,
        step_size_a=0.025,
        temperature_k=300.0,
        burn_in=6,
        stride=2,
        seed=123,
    )
    assert summary.n_frames == 13
    assert summary.ligand_rmsf.shape == (twin.ligand_params.num_atoms,)
    assert summary.pocket_sidechain_rmsf.shape == (twin.pocket_params.num_atoms,)
    assert np.all(np.isfinite(summary.ligand_rmsf))
    assert np.all(np.isfinite(summary.pocket_sidechain_rmsf))
    assert 0.005 < summary.mean_ligand_rmsf < 2.0
    assert math.isfinite(summary.mean_energy_kcal_mol)
    assert 0.0 <= summary.dynamic_stability_score <= 1.0
    assert summary.residence_time_proxy >= 0.0

    for occ_map in (
        summary.hbond_occupancy,
        summary.salt_bridge_occupancy,
        summary.halogen_sigma_hole_occupancy,
        summary.pi_interaction_occupancy,
    ):
        for prob in occ_map.values():
            assert 0.0 <= prob <= 1.0

    for pct in summary.interaction_persistence.values():
        assert 0.0 <= pct <= 100.0

    # Equilibrium twin coordinates must remain untouched
    assert torch.allclose(twin.ligand_params.coords, eq_lig_coords, atol=1e-6)
    assert torch.allclose(twin.pocket_params.coords, eq_rec_coords, atol=1e-6)


def test_guidance_force_fn_and_openmm_adapter() -> None:
    """Verify ``guidance_force_fn`` unbatched/batched float32 forces and ``OpenMMTwinAdapter``."""
    twin = _build_demo_twin()

    x_single = twin.ligand_params.coords.clone()
    f_single = twin.guidance_force_fn(x_single, softcore_lambda=0.80, max_force_norm=20.0)
    assert f_single.shape == x_single.shape
    assert f_single.dtype == torch.float32
    assert bool(torch.isfinite(f_single).all())
    assert float(torch.linalg.vector_norm(f_single, dim=-1).max().item()) <= 20.01

    # Batched [B, M, 3] probe with arbitrary M = 10 generated heavy atoms
    flow_cb = twin.make_flow_guidance_callback()
    x_batch = torch.randn((2, 10, 3), dtype=torch.float32)
    f_batch = flow_cb(x_batch, 0.5)
    assert f_batch.shape == (2, 10, 3)
    assert f_batch.dtype == torch.float32
    assert bool(torch.isfinite(f_batch).all())

    # OpenMM spec export & optional dependency check
    spec = OpenMMTwinAdapter.export_nonbonded_spec(twin)
    total_n = twin.ligand_params.num_atoms + twin.pocket_params.num_atoms
    assert spec["positions_nm"].shape == (total_n, 3)
    assert spec["charges_e"].shape == (total_n,)
    if not OpenMMTwinAdapter.is_available():
        with pytest.raises(OptionalDependencyError) as exc_info:
            OpenMMTwinAdapter.require_openmm()
        assert "openmm" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 6. ESP Grid, Interaction Network, Matplotlib Figure & WebGL HTML/JSON Export
# ---------------------------------------------------------------------------


def test_esp_grid_figure_and_webgl_scene_html_exporters(tmp_path: Path) -> None:
    """Verify 3D ESP grid, Matplotlib figure, TypeScript JSON schema, and WebGL HTML viewer."""
    # Position ligand so NH3+ engages ASP103 salt bridge and aromatic ring faces PHE389
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH3+]CCc1ccc(Br)cc1"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # type: ignore[attr-defined]
    conf = mol.GetConformer(0)
    # Shift NH3+ nitrogen (atom 0) near (-3.1, -1.4, 1.1) adjacent to ASP103 OD1/OD2
    n_pos = conf.GetAtomPosition(0)
    dx, dy, dz = -3.1 - n_pos.x, -1.4 - n_pos.y, 1.1 - n_pos.z
    for idx in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(idx, Point3D(p.x + dx, p.y + dy, p.z + dz))

    twin = ProteinLigandDigitalTwin.from_pdb_and_mol(SYNTHETIC_GPCR_POCKET_PDB, mol)
    twin.relax_complex(steps=35, flexible_sidechains=True)

    # 1. 3D ESP & Electric Field Grid
    grid = compute_electrostatic_potential_grid(twin, grid_spacing=1.2)
    assert grid.esp_kbt_e.dtype == np.float32
    assert grid.electric_field_v_a.dtype == np.float32
    assert grid.electric_field_v_a.shape == (*grid.shape, 3)
    assert np.all(np.isfinite(grid.esp_kbt_e))
    grid_summary = grid.to_summary_dict()
    assert grid_summary["esp_min_kbt_e"] < 0.0  # Anionic ASP103 negative potential well
    assert grid_summary["esp_max_kbt_e"] > 0.0  # Cationic LYS190 positive potential region

    # 2. Typed non-covalent interaction network
    network = extract_typed_interaction_network(twin)
    assert len(network) >= 1
    kinds = {edge.kind for edge in network}
    assert "salt_bridge" in kinds or "hbond" in kinds
    for edge in network:
        d_dict = edge.to_dict()
        assert len(d_dict["start_xyz"]) == 3
        assert len(d_dict["end_xyz"]) == 3
        assert d_dict["distance_angstrom"] > 0.0

    # 3. Publication Matplotlib 4-panel figure
    png_path = tmp_path / "biophys_twin_fig.png"
    fig = render_biophysical_twin_figure(twin, out_path=png_path, dpi=120)
    plt.close(fig)
    assert png_path.is_file() and png_path.stat().st_size > 15_000

    # 4. TypeScript / WebGL JSON Scene Schema
    json_path = tmp_path / "biophys_twin_scene.json"
    scene = export_web_twin_scene_json(twin, out_path=json_path)
    assert json_path.is_file()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["schema_version"] == "geneuclidean.biophysics.web_twin_scene.v1"
    assert reloaded["typescript_interface"] == "BiophysicalTwinSceneV1"
    assert len(reloaded["ligand_atoms"]) == twin.ligand_params.num_atoms
    assert len(reloaded["pocket_atoms"]) == twin.pocket_params.num_atoms
    assert len(reloaded["halogen_sigma_holes"]) == 1
    assert "dg_bind_est" in reloaded["energy_summary"]
    assert scene["metadata"]["units"]["electric_field"] == "V/angstrom"

    # 5. Self-contained Interactive WebGL HTML Viewer
    html_path = export_interactive_twin_html(twin, tmp_path / "biophys_twin_viewer.html")
    assert html_path.is_file()
    html_text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert 'id="twin-webgl-canvas"' in html_text
    assert 'id="biophysical-twin-scene-data"' in html_text
    assert "geneuclidean.biophysics.web_twin_scene.v1" in html_text
    # Ensure 100% offline self-contained viewer (no external CDN <script src="http...">)
    assert "<script src=" not in html_text


def test_biophysics_public_module_alias_consistent() -> None:
    """Top-level ``bp`` lazy exports point to exact implementation classes."""
    assert bp.ProteinLigandDigitalTwin is ProteinLigandDigitalTwin
    assert bp.BiophysicalHamiltonian is BiophysicalHamiltonian
    assert bp.BiophysicalAtomParams is BiophysicalAtomParams
