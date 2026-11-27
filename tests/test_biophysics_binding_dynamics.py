"""Acceptance tests for the rigid-body Langevin binding-dynamics engine.

The tests are split by what they can prove. The mechanics tests use no force field at all,
so a failure there is unambiguously the integrator's. The force-field tests then check that
the native C++ engine samples the *same* potential as the PyTorch reference, which is the
one property that cannot be inferred from a plausible-looking animation.

Everything is built from a synthetic complex written into ``tmp_path`` rather than from the
repository's CASF structures, so the suite does not depend on untracked data.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from geneuclidean.biophysics.binding_dynamics import (
    EIGHT_PI_ETA_WATER_300K,
    MD_ACCEL_FACTOR,
    SIX_PI_ETA_WATER_300K,
    BindingDynamicsConfig,
    _free_top_flow,
    _matrix_to_quat,
    _quat_to_matrix,
    find_escape_direction,
    hydrodynamic_radius,
    ligand_masses,
    make_dynamics_hamiltonian,
    rigid_body_inertia,
    simulate_binding,
)
from geneuclidean.biophysics.forcefield import KB_KCAL_MOL_K

KT_300 = 300.0 * KB_KCAL_MOL_K
KT_RED_300 = MD_ACCEL_FACTOR * KT_300
#: Measured ligand inertia tensor, strongly anisotropic, used by the free-top tests.
DRUG_MOMENTS = torch.tensor([452.1, 1125.1, 1509.9], dtype=torch.float64)


# =======================================================================================
# Mechanics: no force field involved, so failures here are the integrator's alone.
# =======================================================================================


def test_inertia_tensor_matches_the_analytic_dumbbell() -> None:
    """Two unit masses at x = +-d give I = diag(0, 2md^2, 2md^2) about the midpoint."""
    d = 2.0
    coords = torch.tensor([[-d, 0.0, 0.0], [d, 0.0, 0.0]])
    masses = torch.tensor([1.0, 1.0])
    com, moments, _ = rigid_body_inertia(coords, masses)

    assert torch.allclose(com, torch.zeros(3), atol=1e-6)
    assert float(moments[0]) == pytest.approx(0.0, abs=2e-3)  # clamped at 1e-3
    assert float(moments[1]) == pytest.approx(8.0, rel=1e-5)
    assert float(moments[2]) == pytest.approx(8.0, rel=1e-5)


def test_principal_axes_are_a_proper_rotation_not_a_reflection() -> None:
    """``eigh`` may return det = -1; feeding that to the quaternion mirrors the ligand."""
    torch.manual_seed(0)
    for _ in range(25):
        coords = torch.randn(9, 3)
        masses = torch.rand(9) + 0.5
        _, _, axes = rigid_body_inertia(coords, masses)
        assert float(torch.det(axes)) == pytest.approx(1.0, abs=1e-4)


def test_matrix_to_quat_inverts_quat_to_matrix() -> None:
    """Round-trip over random rotations, including the near-trace-minus-one branches."""
    torch.manual_seed(1)
    for _ in range(200):
        q = torch.randn(4)
        q = q / torch.linalg.vector_norm(q)
        rot = _quat_to_matrix(q)
        back = _quat_to_matrix(_matrix_to_quat(rot))
        assert torch.allclose(rot, back, atol=1e-5)


def test_rigid_body_frame_reconstructs_the_input_conformer() -> None:
    """``body_ref`` lives in the principal-axis basis, so the attitude must be ``axes``.

    Starting from the identity quaternion instead silently re-orients the ligand, which is
    how a 1e7 kcal/mol/A clash gets introduced on the very first step.
    """
    torch.manual_seed(2)
    coords = torch.randn(14, 3) * 3.0 + 20.0
    masses = torch.rand(14) * 10 + 1.0
    com, _, axes = rigid_body_inertia(coords, masses)
    body_ref = (coords - com) @ axes

    rebuilt = body_ref @ _quat_to_matrix(_matrix_to_quat(axes)).T + com
    assert float((rebuilt - coords).abs().max()) < 1e-4

    # The identity quaternion is wrong unless the eigenbasis happens to be the identity.
    wrong = body_ref @ torch.eye(3) + com
    assert float((wrong - coords).abs().max()) > 1e-2


def test_free_top_conserves_angular_momentum_and_energy() -> None:
    """DLM R-splitting is exactly angular-momentum conserving and second order.

    Both properties are invisible in float32 (round-off dominates at these step counts), so
    the test runs in float64 -- which is also how the engine stores its state.
    """
    omega0 = torch.tensor([0.30, 0.20, 0.10], dtype=torch.float64)
    results = {}
    for dt in (0.002, 0.004):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        ang_mom = DRUG_MOMENTS * omega0
        e0 = 0.5 * float((ang_mom * ang_mom / DRUG_MOMENTS).sum())
        l0 = float(torch.linalg.vector_norm(_quat_to_matrix(q) @ ang_mom))
        worst_e = worst_l = 0.0
        for _ in range(int(4.0 / dt)):  # 4 ps
            q, ang_mom = _free_top_flow(q, ang_mom, DRUG_MOMENTS, dt)
            e = 0.5 * float((ang_mom * ang_mom / DRUG_MOMENTS).sum())
            lab = float(torch.linalg.vector_norm(_quat_to_matrix(q) @ ang_mom))
            worst_e = max(worst_e, abs(e - e0) / e0)
            worst_l = max(worst_l, abs(lab - l0) / l0)
        results[dt] = (worst_e, worst_l)
        assert float(torch.linalg.vector_norm(q)) == pytest.approx(1.0, abs=1e-12)

    # |L| is conserved to round-off, not merely to truncation order.
    assert results[0.002][1] < 1e-11
    assert results[0.004][1] < 1e-11
    # Energy drift is second order: halving dt must cut it by ~4, not ~2.
    ratio = results[0.004][0] / max(results[0.002][0], 1e-300)
    assert 3.0 < ratio < 5.0, f"expected 2nd-order scaling, got x{ratio:.2f}"


@pytest.mark.slow
def test_rotational_thermostat_reaches_equipartition_per_axis() -> None:
    """Each body-frame rotational degree of freedom must hold (1/2)kT.

    The Ornstein-Uhlenbeck kernel acts on angular *momentum* with per-axis rates
    ``zeta_r / I_a`` and an axis-independent noise strength. Getting that the wrong way
    round -- anisotropic noise, isotropic rate -- still looks like tumbling but thermostats
    the three axes at three different temperatures, which this test would catch.
    """
    gen = torch.Generator().manual_seed(5)
    zeta_r = 1.36e5
    dt = 0.002
    rates = zeta_r / DRUG_MOMENTS
    c1 = torch.exp(-rates * dt)
    c2 = torch.sqrt(1.0 - c1 * c1)
    sigma = torch.sqrt(DRUG_MOMENTS * KT_RED_300)

    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    ang_mom = torch.zeros(3, dtype=torch.float64)
    acc = torch.zeros(3, dtype=torch.float64)
    # At zeta_r / I_a = [300.8, 120.9, 90.1] ps^-1 and dt = 0.002 ps, gamma_a * dt >= 0.180,
    # so the slowest axis decorrelates in ~2.8 steps; n = 20,000 spans > 7,200 independent
    # correlation times (SE ~ 1.66%, well inside rel=0.05) while keeping --cov runtime fast
    # enough for CI's shared 20-minute ubuntu-latest budget.
    n = 20000
    noise_all = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    for step_i in range(n):
        q, ang_mom = _free_top_flow(q, ang_mom, DRUG_MOMENTS, 0.5 * dt)
        ang_mom = c1 * ang_mom + c2 * sigma * noise_all[step_i]
        q, ang_mom = _free_top_flow(q, ang_mom, DRUG_MOMENTS, 0.5 * dt)
        acc += ang_mom * ang_mom / (2.0 * DRUG_MOMENTS) / MD_ACCEL_FACTOR
    acc /= n

    target = 0.5 * KT_300
    for a in range(3):
        assert float(acc[a]) == pytest.approx(target, rel=0.05), (
            f"axis {a} holds {float(acc[a]):.5f} kcal/mol, expected {target:.5f}"
        )


@pytest.mark.slow
def test_translational_thermostat_reaches_equipartition() -> None:
    """The exact OU kernel on the centre-of-mass velocity gives <KE> = (3/2)kT."""
    gen = torch.Generator().manual_seed(7)
    mass, gamma, dt = 333.0, 13.1, 0.002
    c1 = math.exp(-gamma * dt)
    c2 = math.sqrt(1.0 - c1 * c1)
    vel = torch.zeros(3, dtype=torch.float64)
    acc, n = 0.0, 200000
    noise_scale = c2 * math.sqrt(KT_RED_300 / mass)
    noise_all = torch.randn((n, 3), generator=gen, dtype=torch.float64)
    for step_i in range(n):
        vel = c1 * vel + noise_scale * noise_all[step_i]
        acc += 0.5 * mass * float((vel * vel).sum()) / MD_ACCEL_FACTOR
    assert acc / n == pytest.approx(1.5 * KT_300, rel=0.03)


def test_hydrodynamic_radius_of_a_uniform_sphere() -> None:
    """A shell of mass at radius a has R_g = a, so the equivalent sphere is a*sqrt(5/3)."""
    pts = torch.tensor(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    masses = torch.ones(6)
    assert hydrodynamic_radius(pts, masses) == pytest.approx(math.sqrt(5.0 / 3.0), rel=1e-6)


def test_stokes_prefactors_match_water_viscosity() -> None:
    """6*pi*eta and 8*pi*eta for water at 300 K, in amu/(A*ps).

    eta = 0.851 mPa*s; 1 Pa*s = 6.0221e4 amu/(A*ps), so eta = 51.25 amu/(A*ps).
    """
    eta = 0.851e-3 * 6.0221e4
    assert pytest.approx(6 * math.pi * eta, rel=2e-3) == SIX_PI_ETA_WATER_300K
    assert pytest.approx(8 * math.pi * eta, rel=2e-3) == EIGHT_PI_ETA_WATER_300K


# =======================================================================================
# Force field: a synthetic complex, used to pin the native engine to the reference.
# =======================================================================================

_POCKET_PDB = """\
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


@pytest.fixture(scope="module")
def synthetic_complex(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    """A small ligand in a crude three-residue pocket, built without any repo data."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from geneuclidean.biophysics.forcefield import (
        parameterize_ligand_biophysics,
        parameterize_pocket_biophysics,
    )

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore[attr-defined]

    pdb = tmp_path_factory.mktemp("pocket") / "pocket.pdb"
    pdb.write_text(_POCKET_PDB)

    lig = parameterize_ligand_biophysics(mol)
    rec = parameterize_pocket_biophysics(str(pdb))
    return lig, rec


def _native_or_skip() -> Any:
    """Import the native bridge, skipping the test when it cannot be compiled."""
    nd = pytest.importorskip("geneuclidean.biophysics._native_dynamics")
    try:
        nd.build_native_library()
    except Exception as exc:  # pragma: no cover - depends on the toolchain
        pytest.skip(f"native engine unavailable: {exc}")
    return nd


def test_make_dynamics_hamiltonian_widens_the_switching_window() -> None:
    """The shipped 10-12 A window puts a spurious attractive shell where the approach is.

    Switching a potential rather than a force leaves a 0.543 kT/A artifact near 11 A, which
    a trajectory would read as a genuine funnel; 20-28 A brings that down to 0.048 kT/A and
    costs nothing, because the pair tensor is dense either way.
    """
    ham = make_dynamics_hamiltonian()
    assert ham.switch_on == 20.0
    assert ham.nonbonded_cutoff == 28.0
    # Still overridable.
    assert make_dynamics_hamiltonian(switch_on=10.0, nonbonded_cutoff=12.0).switch_on == 10.0


def test_find_escape_direction_threads_open_channel_of_buried_cup_pocket() -> None:
    """On a buried cup pocket whose centroid lies on the -y side while the only open mouth
    is along +x, the naive pocket-centroid-to-ligand vector points straight into the +y wall,
    whereas ``find_escape_direction`` selects the +x open channel.
    """
    lig_coords = torch.tensor(
        [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0]],
        dtype=torch.float32,
    )

    # Fibonacci shell of radius 4.6 A around origin with a single 34-deg mouth along +x
    idx = torch.arange(180, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * idx / 180.0)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    unit_pts = torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)],
        dim=1,
    ).to(dtype=torch.float32)
    keep_wall = unit_pts[:, 0] < math.cos(math.radians(34.0))
    cup_wall = 4.6 * unit_pts[keep_wall]

    # Add a dense protein core behind -y so pocket centroid shifts strongly toward -y,
    # making (lig_com - rec_com) point along +y right into the closed +y cup wall.
    core_grid = torch.tensor(
        [[x, y, z] for x in (-2.0, 0.0, 2.0) for y in (-9.0, -7.5, -6.0) for z in (-2.0, 0.0, 2.0)],
        dtype=torch.float32,
    )
    rec_coords = torch.cat([cup_wall, core_grid], dim=0)

    naive_dir = lig_coords.mean(0) - rec_coords.mean(0)
    naive_dir = naive_dir / torch.linalg.vector_norm(naive_dir)

    esc = find_escape_direction(lig_coords, rec_coords)
    assert esc.dtype == torch.float32
    assert float(torch.linalg.vector_norm(esc)) == pytest.approx(1.0, abs=1e-5)

    # Escape vector must align with the +x open mouth, far from the blocked +y naive vector
    assert float(esc[0]) > 0.90, f"expected +x channel escape, got {esc.tolist()}"
    assert float(naive_dir[1]) > 0.85, f"fixture sanity: naive vector {naive_dir.tolist()}"

    ts = torch.arange(2.0, 25.0, 0.5, dtype=torch.float32).unsqueeze(-1)
    com = lig_coords.mean(0)
    naive_clearance = float(torch.cdist(com + ts * naive_dir, rec_coords).min())
    esc_clearance = float(torch.cdist(com + ts * esc, rec_coords).min())
    assert naive_clearance < 1.2
    assert esc_clearance > 2.3


def test_native_energy_matches_the_pytorch_reference(synthetic_complex: tuple) -> None:
    """The whole point of the native engine: the same potential, not a similar one.

    The reference runs in float32 and the engine in float64, so exact equality is
    impossible; 1e-4 relative is comfortably above that noise floor and far below anything
    that would change the dynamics.
    """
    nd = _native_or_skip()
    lig, rec = synthetic_complex
    ham = make_dynamics_hamiltonian()
    eng = nd.NativeEngine.build(lig, rec, ham)
    try:
        # The mirrored descreening integral must reproduce the reference Born radii.
        assert eng.born_check < 2e-4

        u_cpp, _ = eng.energy(eng.quat0, eng.com0)
        u_py = float(
            ham.potential_energy(
                lig.coords.detach(), rec.coords.detach(), lig, rec, softcore_lambda=1.0
            )
        )
        assert u_cpp == pytest.approx(u_py, rel=1e-4, abs=1e-4)
    finally:
        eng.close()


def test_native_energy_matches_away_from_the_input_pose(synthetic_complex: tuple) -> None:
    """Agreement at one pose proves little; the sampler visits displaced ones."""
    nd = _native_or_skip()
    lig, rec = synthetic_complex
    ham = make_dynamics_hamiltonian()
    eng = nd.NativeEngine.build(lig, rec, ham)
    try:
        rng = np.random.default_rng(0)
        q0 = eng.quat0
        for trial in range(5):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            ang = rng.uniform(0.0, np.pi)
            dq = np.concatenate([[np.cos(ang / 2)], axis * np.sin(ang / 2)])
            quat = np.array(
                [
                    dq[0] * q0[0] - dq[1] * q0[1] - dq[2] * q0[2] - dq[3] * q0[3],
                    dq[0] * q0[1] + dq[1] * q0[0] + dq[2] * q0[3] - dq[3] * q0[2],
                    dq[0] * q0[2] - dq[1] * q0[3] + dq[2] * q0[0] + dq[3] * q0[1],
                    dq[0] * q0[3] + dq[1] * q0[2] - dq[2] * q0[1] + dq[3] * q0[0],
                ]
            )
            quat /= np.linalg.norm(quat)
            com = eng.com0 + rng.normal(size=3) * (0.4 + 0.5 * trial)

            u_cpp, _ = eng.energy(quat, com)
            coords = torch.from_numpy(eng.pose(quat, com).astype(np.float32))
            u_py = float(
                ham.potential_energy(coords, rec.coords.detach(), lig, rec, softcore_lambda=1.0)
            )
            assert u_cpp == pytest.approx(u_py, rel=2e-4, abs=1e-3), (
                f"trial {trial}: native {u_cpp} vs reference {u_py}"
            )
    finally:
        eng.close()


def test_native_finite_difference_forces_match_autograd(synthetic_complex: tuple) -> None:
    """Central differences on the six DOF must reproduce the autograd gradient.

    This is what licenses using finite differences instead of hand-differentiating the
    Generalized Born chain rule, the donor axes, the ring normals and the SASA.
    """
    nd = _native_or_skip()
    lig, rec = synthetic_complex
    ham = make_dynamics_hamiltonian()
    eng = nd.NativeEngine.build(lig, rec, ham)
    try:
        f_cpp, t_cpp = eng.forces(eng.quat0, eng.com0)

        pose = eng.pose(eng.quat0, eng.com0)
        coords = torch.from_numpy(pose.astype(np.float32)).requires_grad_(True)
        u = ham.potential_energy(coords, rec.coords.detach(), lig, rec, softcore_lambda=1.0)
        (grad,) = torch.autograd.grad(u, coords)
        f_atoms = -grad.detach().numpy().astype(np.float64)
        f_ref = f_atoms.sum(0)
        t_ref = np.cross(pose - eng.com0, f_atoms).sum(0)

        assert np.linalg.norm(f_cpp - f_ref) / max(np.linalg.norm(f_ref), 1e-9) < 1e-3
        assert np.linalg.norm(t_cpp - t_ref) / max(np.linalg.norm(t_ref), 1e-9) < 1e-3
    finally:
        eng.close()


def test_simulate_binding_rejects_an_unknown_backend(synthetic_complex: tuple) -> None:
    lig, rec = synthetic_complex
    with pytest.raises(ValueError, match="backend must be"):
        simulate_binding(lig, rec, make_dynamics_hamiltonian(), backend="gpu")


@pytest.mark.slow
def test_short_trajectory_is_thermal_and_sterically_sane(synthetic_complex: tuple) -> None:
    """End-to-end: the recorded run must be at temperature and must not tunnel.

    Kept short, so the temperature bound is loose -- a 3-DOF instantaneous temperature has
    an enormous per-frame spread and only the mean over many steps is meaningful.
    """
    _native_or_skip()
    lig, rec = synthetic_complex
    cfg = BindingDynamicsConfig(
        release_steps=600, record_every=20, launch_distance_a=3.0, stage_max_steps=400
    )
    traj = simulate_binding(lig, rec, make_dynamics_hamiltonian(), config=cfg, backend="native")

    assert traj.frames, "no frames recorded"
    assert traj.diagnostics["softcore_lambda"] == 1.0
    assert traj.diagnostics["force_clipping"] == "none"

    masses = ligand_masses(lig)
    assert traj.diagnostics["total_mass_amu"] == pytest.approx(float(masses.sum()), rel=1e-3)

    # Thermal: the mean kinetic energies sit near (3/2)kT for each triple of DOF.
    expected = traj.diagnostics["expected_ke_per_3dof_kcal_mol"]
    assert traj.diagnostics["mean_ke_trans_kcal_mol"] == pytest.approx(expected, rel=0.6)
    assert traj.diagnostics["mean_ke_rot_kcal_mol"] == pytest.approx(expected, rel=0.6)

    # Steric: nothing may pass through an atom. Below ~1.0 A is unphysical overlap.
    assert traj.diagnostics["closest_contact_a"] > 1.0

    # Smooth: no frame-to-frame teleporting, which would betray an unstable timestep.
    coords = np.array([f.coords for f in traj.frames])
    rms = np.sqrt(((coords[1:] - coords[:-1]) ** 2).sum(-1).mean(-1))
    assert float(rms.max()) < 5.0, f"largest per-frame RMS jump {float(rms.max()):.2f} A"


def test_trajectory_serializes_with_a_schema_version(synthetic_complex: tuple) -> None:
    """The viewer consumes ``to_dict``; its shape is part of the contract."""
    _native_or_skip()
    lig, rec = synthetic_complex
    cfg = BindingDynamicsConfig(
        release_steps=60, record_every=20, launch_distance_a=2.0, stage_max_steps=60
    )
    traj = simulate_binding(lig, rec, make_dynamics_hamiltonian(), config=cfg, backend="native")
    payload = traj.to_dict()

    assert payload["schema_version"] == "geneuclidean.biophysics.binding_trajectory.v1"
    assert payload["frames"]
    first = payload["frames"][0]
    for key in (
        "step",
        "time_ps",
        "com",
        "quat",
        "coords",
        "u_total",
        "e_coulomb",
        "dg_gb_solvation",
        "e_vdw",
        "e_hbond",
        "temp_trans_k",
        "temp_rot_k",
        "com_displacement_a",
        "min_contact_a",
    ):
        assert key in first, f"missing frame field {key}"
    assert len(first["quat"]) == 4
    assert len(first["com"]) == 3
    assert len(first["coords"]) == int(lig.coords.shape[0])
    assert "contract" in payload["diagnostics"]


def test_native_library_source_is_packaged() -> None:
    """The C++ source must ship, or the engine cannot be built from an installed wheel."""
    import geneuclidean.biophysics as bio

    src = Path(bio.__file__).parent / "_native" / "binding_dynamics.cpp"
    assert src.is_file(), f"missing native source at {src}"


def test_cxx_unavailable_degrades_gracefully_and_auto_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_complex: tuple,
) -> None:
    """When C++ compilation fails or CXX is missing, build raises and backend='auto' falls back."""
    from geneuclidean.biophysics._native_dynamics import build_native_library, native_available

    # 1. Failing compiler exit code -> RuntimeError with compiler command diagnostics
    false_bin = shutil.which("false")
    if false_bin is not None:
        monkeypatch.setenv("CXX", false_bin)
        with pytest.raises(RuntimeError, match="failed to compile"):
            build_native_library(tmp_path / "fail_compile.so", force=True)

    # 2. Nonexistent CXX binary + unbuilt GENEUCLIDEAN_NATIVE_LIB -> clean fallback to Python
    missing_lib = tmp_path / "unbuilt_libgexdyn.so"
    monkeypatch.setenv("CXX", "/nonexistent/gex_missing_cxx_compiler")
    monkeypatch.setenv("GENEUCLIDEAN_NATIVE_LIB", str(missing_lib))

    with pytest.raises((RuntimeError, FileNotFoundError)):
        build_native_library(missing_lib, force=True)

    assert native_available() is False

    lig, rec = synthetic_complex
    cfg = BindingDynamicsConfig(
        release_steps=4,
        record_every=2,
        launch_distance_a=1.0,
        stage_max_steps=4,
    )
    traj = simulate_binding(lig, rec, make_dynamics_hamiltonian(), config=cfg, backend="auto")
    assert len(traj.frames) >= 2
    assert "python" in str(traj.diagnostics["backend"]).lower()


def _find_empp() -> Path | None:
    found = shutil.which("em++")
    if found is not None:
        return Path(found)
    emsdk_candidate = Path.home() / "emsdk" / "upstream" / "emscripten" / "em++"
    return emsdk_candidate if emsdk_candidate.is_file() else None


_EMPP_PATH = _find_empp()
_NODE_PATH = shutil.which("node")


@pytest.mark.skipif(
    _EMPP_PATH is None or _NODE_PATH is None,
    reason="Emscripten (em++) and Node.js required for WebAssembly parity verification",
)
def test_wasm_module_matches_native_engine_via_instantiate_wasm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_complex: tuple,
) -> None:
    """Compile _native/binding_dynamics.cpp to WASM and verify parity via Module.instantiateWasm."""
    nd = _native_or_skip()
    assert _EMPP_PATH is not None and _NODE_PATH is not None

    monkeypatch.setenv("PATH", f"{_EMPP_PATH.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    wasm_dir = tmp_path / "wasm_out"
    js_path = nd.build_wasm_module(wasm_dir, optimize="-O2")
    wasm_path = js_path.with_suffix(".wasm")
    assert js_path.is_file() and js_path.stat().st_size > 1_000
    assert wasm_path.is_file() and wasm_path.stat().st_size > 10_000

    lig, rec = synthetic_complex
    ham = make_dynamics_hamiltonian()
    blob = nd.pack_engine_blob(lig, rec, ham)
    blob_path = tmp_path / "engine.bin"
    blob_path.write_bytes(blob)

    eng = nd.NativeEngine.build(lig, rec, ham)
    try:
        quat = np.ascontiguousarray(eng.quat0, dtype=np.float64)
        com = np.ascontiguousarray(eng.com0 + np.array([0.35, -0.25, 0.40]), dtype=np.float64)
        u_cpp, terms_cpp = eng.energy(quat, com)
        f_cpp, t_cpp = eng.forces(quat, com)
    finally:
        eng.close()

    pose_json = json.dumps({"quat": quat.tolist(), "com": com.tolist()})
    harness_js = tmp_path / "verify_wasm.cjs"
    harness_js.write_text(
        """\
const fs = require("fs");
const vm = require("vm");

const [jsPath, wasmPath, blobPath, poseJson] = process.argv.slice(2);
const wasmBytes = fs.readFileSync(wasmPath);
const blobBytes = fs.readFileSync(blobPath);
const pose = JSON.parse(poseJson);

globalThis.self = globalThis;
globalThis.location = { href: "file://" + jsPath };
vm.runInThisContext(fs.readFileSync(jsPath, "utf8") + "\\n;globalThis.__GexDyn = GexDyn;");

(async () => {
  const mod = await globalThis.__GexDyn({
    instantiateWasm(imports, successCallback) {
      WebAssembly.instantiate(wasmBytes, imports).then((res) => {
        successCallback(res.instance, res.module);
      });
      return {};
    },
  });

  const version = mod._gex_version();
  const blobPtr = mod._malloc(blobBytes.length);
  mod.HEAPU8.set(blobBytes, blobPtr);
  const handle = mod._gex_create_packed(blobPtr, blobBytes.length);
  mod._free(blobPtr);
  if (!handle) {
    throw new Error("gex_create_packed returned 0");
  }

  const qPtr = mod._malloc(4 * 8);
  const cPtr = mod._malloc(3 * 8);
  const termsPtr = mod._malloc(8 * 8);
  const fPtr = mod._malloc(3 * 8);
  const tPtr = mod._malloc(3 * 8);

  mod.HEAPF64.set(pose.quat, qPtr >> 3);
  mod.HEAPF64.set(pose.com, cPtr >> 3);

  const uTotal = mod._gex_energy(handle, qPtr, cPtr, termsPtr);
  mod._gex_forces(handle, qPtr, cPtr, 1e-5, 1e-5, fPtr, tPtr);

  const terms = Array.from(mod.HEAPF64.subarray(termsPtr >> 3, (termsPtr >> 3) + 8));
  const force = Array.from(mod.HEAPF64.subarray(fPtr >> 3, (fPtr >> 3) + 3));
  const torque = Array.from(mod.HEAPF64.subarray(tPtr >> 3, (tPtr >> 3) + 3));

  mod._free(qPtr);
  mod._free(cPtr);
  mod._free(termsPtr);
  mod._free(fPtr);
  mod._free(tPtr);
  mod._gex_destroy(handle);

  process.stdout.write(JSON.stringify({ version, uTotal, terms, force, torque }) + "\\n");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [_NODE_PATH, str(harness_js), str(js_path), str(wasm_path), str(blob_path), pose_json],
        capture_output=True,
        text=True,
        check=True,
    )
    wasm_out = json.loads(proc.stdout.strip())
    assert wasm_out["version"] == 7
    assert float(wasm_out["uTotal"]) == pytest.approx(u_cpp, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(wasm_out["terms"], terms_cpp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(wasm_out["force"], f_cpp, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(wasm_out["torque"], t_cpp, rtol=1e-9, atol=1e-9)
