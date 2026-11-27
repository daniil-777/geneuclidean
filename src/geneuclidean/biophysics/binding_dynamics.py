"""Rigid-body Langevin dynamics of ligand association with a frozen pocket.

Produces a time-resolved trajectory of a ligand approaching a binding pocket, entering it,
fluctuating under thermal noise and settling -- suitable for animation and for measuring
the residence of individual contacts.

Why rigid body, and not all-atom flexible
-----------------------------------------
:mod:`geneuclidean.biophysics.forcefield` has **no torsion or dihedral term**:
``compute_internal_restraint_energy`` is bond lengths plus cosine-harmonic angles and
nothing else. An all-atom flexible ligand integrated for thousands of steps under
bond-and-angle restraints alone has nothing holding its dihedrals in chemically sensible
positions, so it slowly melts into a geometry no chemist would accept while the animation
still looks perfectly smooth. Integrating the **six rigid-body degrees of freedom**
sidesteps that entirely and is established practice for association dynamics (BrownDye,
SDA). Internal geometry is frozen at the input conformer.

Design decisions forced by measurement
--------------------------------------
Each of these was measured on this force field, not assumed:

1. **No force clipping.** ``_smooth_clip_vectors`` in
   :mod:`geneuclidean.biophysics.protein_ligand_twin` rescales each atom's force by
   ``rsqrt(1 + |F|^2/c^2)``. That is not the gradient of any potential, so a dynamics using
   it has **no Boltzmann stationary distribution**. This module takes the raw
   ``-torch.autograd.grad``. Stability comes from the timestep instead -- and a rigid body
   has no bonded modes to resolve, so the limit is loose: the Lennard-Jones pair curvature
   for the repo's carbon parameters is 248 kcal/mol/A^2 at 0.75 sigma, which for a 350 amu
   ligand is a hard limit near 116 fs, and roughly 37 fs even with ten simultaneous
   clashing contacts. The 2 fs default is far inside that. The familiar "1 fs because of
   C-H stretches" argument does **not** apply here and must not be quoted as if it did.
2. **One Hamiltonian only.** ``sample_thermal_ensemble`` hardcodes
   ``softcore_lambda=0.95`` while its guidance force defaults to ``0.85``, so the motion
   and the reported energy come from different potentials. Everything here runs and reports
   at ``softcore_lambda=1.0``.
3. **Widen the nonbonded switch before integrating.** Switching a *potential* rather than a
   force makes the ``-U dS/dr`` term dominate near the cutoff, so the default 10-12 A
   window carries a spurious attractive shell: the switched/unswitched force ratio peaks at
   **5.23 at r = 11.1 A**. That ratio cannot be driven to 1 by any window width -- it is
   O(1) by construction, because the unswitched force vanishes faster than the switching
   term -- so the honest gate is the *absolute* spurious force, which falls from 0.543
   kT/A (10-12 A) to **0.048 kT/A (20-28 A)**. Use :func:`make_dynamics_hamiltonian`, which
   widens the window to 20-28 A. This costs nothing: the code has no neighbour list, the
   pair tensor is dense either way, and the cutoff only zeroes energies.
4. **Exact free-top rotation, not a gyroscopic half-kick.** Putting
   ``-omega x (I omega)`` inside the B half-kicks makes the kick depend on the variable it
   updates, so the splitting is not symmetric and the scheme is **first order**: measured
   relative energy drift on a torque-free asymmetric top scales linearly in the timestep.
   The Dullweber-Leimkuhler-McLachlan symmetric R-splitting of the free top is second order
   and conserves the lab-frame angular momentum to round-off; measured drift is four to
   five orders of magnitude smaller at the same timestep. This module uses DLM.
5. **Rotational anisotropy belongs in the friction, not the noise.** For an isotropic
   hydrodynamic drag ``zeta_r`` the noise amplitude ``sqrt(2 zeta_r kT)`` is the same on
   every axis; what differs per axis is the *rate* ``zeta_r / I_a``. The thermostat is
   therefore an exact Ornstein-Uhlenbeck kernel on the **body-frame angular momentum**,
   with three different rates and a common noise strength. Those rates reach order
   ``1 / dt``, which is why the exact kernel is mandatory rather than merely tidier.

Residual artifacts that cannot be fixed here, only disclosed
------------------------------------------------------------
* The H-bond term carries its **own** switch over 3.8-4.8 A, with the same pathology: a
  spurious excess of about 0.40 kT/A per aligned donor-acceptor pair near 4.0 A. Widening
  it would extend an unphysical attraction further out, so it is left alone.
* The H-bond angular term is clamped at short range, which zeroes the H-bond force inside
  ``r = 2.476 A``. Crossing that radius steps the force discontinuously -- about 72
  kcal/mol/A for a neutral pair and up to **139 kcal/mol/A for a salt bridge**. Frames that
  cross it are integrating through a C^0 kink in the force.

What this is, stated precisely
------------------------------
This is **exact Boltzmann sampling of a fixed empirical scoring function at a nominal
300 K**, which is internally consistent statistical mechanics on that function. It is
*not* a simulation of real solvated binding kinetics, for three reasons worth keeping in
view: the potential already contains a solvation *free* energy (``dg_gb``), so
Boltzmann-sampling it double-counts solvent entropy and ``temperature_k`` is a sampling
parameter rather than a thermodynamic temperature; the receptor is rigid; and that scoring
function's measured correlation with experimental affinity is Pearson R = 0.04 on
CASF-2016 (see ``scripts/run_biophysics_casf_benchmark.py``). Treat the trajectory as a
visualisation of the model's own energy landscape, never as evidence about a real ligand.

Units
-----
Angstrom, picosecond, atomic mass unit, kcal/mol, Kelvin. The one conversion that matters:
``a[A/ps^2] = 418.4 * F[kcal/mol/A] / m[amu]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from geneuclidean.biophysics.forcefield import KB_KCAL_MOL_K

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geneuclidean.biophysics.forcefield import (
        BiophysicalAtomParams,
        BiophysicalHamiltonian,
    )

__all__ = [
    "EIGHT_PI_ETA_WATER_300K",
    "MD_ACCEL_FACTOR",
    "SIX_PI_ETA_WATER_300K",
    "BindingDynamicsConfig",
    "BindingTrajectory",
    "RigidBodyState",
    "TrajectoryFrame",
    "find_escape_direction",
    "hydrodynamic_radius",
    "ligand_masses",
    "make_dynamics_hamiltonian",
    "rigid_body_inertia",
    "simulate_binding",
]

#: 1 kcal/mol = 418.4 amu*A^2/ps^2, so a[A/ps^2] = 418.4 * F[kcal/mol/A] / m[amu].
MD_ACCEL_FACTOR = 418.4

# Water viscosity at 300 K is 0.851 mPa*s. Converting: 1 Pa*s = 1 kg/(m*s) and
# 1 kg = 6.0221e26 amu, 1 m = 1e10 A, 1 s = 1e12 ps, so 1 Pa*s = 6.0221e4 amu/(A*ps)
# and eta = 51.25 amu/(A*ps).
#: Stokes translational drag coefficient prefactor: ``zeta_t = SIX_PI_ETA * a``.
SIX_PI_ETA_WATER_300K = 966.1
#: Stokes-Einstein-Debye rotational drag prefactor: ``zeta_r = EIGHT_PI_ETA * a**3``.
EIGHT_PI_ETA_WATER_300K = 1288.1


def make_dynamics_hamiltonian(**kwargs: Any) -> BiophysicalHamiltonian:
    """Build a :class:`BiophysicalHamiltonian` with a switching window safe for dynamics.

    The shipped default window (``switch_on=10``, ``nonbonded_cutoff=12``) leaves a
    spurious attractive force of 0.543 kT/A near 11 A, which a trajectory would read as a
    real funnel. Widening to 20-28 A drops that to 0.048 kT/A and is computationally free,
    because the pair tensor is dense regardless of the cutoff.

    Args:
        **kwargs: Forwarded to :class:`BiophysicalHamiltonian`. ``switch_on`` and
            ``nonbonded_cutoff`` default to the widened values but may be overridden.

    Returns:
        A Hamiltonian suitable for integrating, rather than only for scoring a fixed pose.
    """
    from geneuclidean.biophysics.forcefield import BiophysicalHamiltonian

    kwargs.setdefault("switch_on", 20.0)
    kwargs.setdefault("nonbonded_cutoff", 28.0)
    return BiophysicalHamiltonian(**kwargs)


@dataclass
class BindingDynamicsConfig:
    """Parameters of a rigid-body Langevin association run.

    Attributes:
        temperature_k: Nominal sampling temperature.
        timestep_ps: Integration timestep. See module docstring: the rigid-body stability
            limit is tens of femtoseconds, so 2 fs is conservative.
        hydrodynamic_radius_a: Equivalent-sphere radius used for Stokes friction. ``None``
            derives it from the ligand's own mass-weighted radius of gyration.
        gamma_trans: Translational collision frequency, 1/ps. ``None`` derives it from
            Stokes drag, ``zeta_t / M`` with ``zeta_t = 6 pi eta a``.
        zeta_rot: Isotropic rotational drag, amu*A^2/ps. ``None`` derives it from
            Stokes-Einstein-Debye, ``8 pi eta a^3``. Per-axis rates are ``zeta_rot / I_a``.
        launch_distance_a: How far along the escape channel to stage the ligand before
            release. 8.0 is the honest edge of super-thermal electrostatic attraction for
            this force field: a unit charge pair feels 0.415 kT/A at 8 A, giving a Peclet
            number near 1.25 over a 3 A step.
        stage_pull_velocity: Steered-MD pull speed of the restraint centre, A/ps.
        stage_spring_k: Steered-MD restraint stiffness, kcal/mol/A^2.
        stage_max_steps: Safety cap on staging steps.
        release_steps: Free Langevin steps recorded as the trajectory.
        record_every: Record one frame per this many release steps.
        seed: RNG seed. Trajectories are reproducible for a given seed.
    """

    temperature_k: float = 300.0
    timestep_ps: float = 0.002
    hydrodynamic_radius_a: float | None = None
    gamma_trans: float | None = None
    zeta_rot: float | None = None
    launch_distance_a: float = 8.0
    stage_pull_velocity: float = 4.0
    stage_spring_k: float = 60.0
    stage_max_steps: int = 4000
    release_steps: int = 12000
    record_every: int = 20
    seed: int = 7


@dataclass
class RigidBodyState:
    """Instantaneous rigid-body state.

    Attributes:
        com: ``[3]`` centre of mass, Angstrom.
        quat: ``[4]`` unit quaternion ``(w, x, y, z)`` mapping body frame to lab frame.
        vel: ``[3]`` centre-of-mass velocity, A/ps.
        ang_mom: ``[3]`` **body-frame** angular momentum, amu*A^2/ps. Stored rather than
            the angular velocity because both the free-top flow and the thermostat are
            exact in this variable.
    """

    com: Tensor
    quat: Tensor
    vel: Tensor
    ang_mom: Tensor


@dataclass
class TrajectoryFrame:
    """One recorded frame."""

    step: int
    time_ps: float
    com: list[float]
    quat: list[float]
    coords: list[list[float]]
    u_total: float
    e_coulomb: float
    dg_gb_solvation: float
    e_vdw: float
    e_hbond: float
    e_halogen: float
    e_cation_pi: float
    e_pi_stacking: float
    dg_hydrophobic: float
    temp_trans_k: float
    temp_rot_k: float
    com_displacement_a: float
    min_contact_a: float


@dataclass
class BindingTrajectory:
    """A recorded association trajectory plus the diagnostics that make it checkable."""

    frames: list[TrajectoryFrame] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for a viewer."""
        return {
            "schema_version": "geneuclidean.biophysics.binding_trajectory.v1",
            "diagnostics": self.diagnostics,
            "frames": [
                {
                    "step": f.step,
                    "time_ps": round(f.time_ps, 5),
                    "com": [round(v, 4) for v in f.com],
                    "quat": [round(v, 5) for v in f.quat],
                    "coords": [[round(c, 3) for c in xyz] for xyz in f.coords],
                    "u_total": round(f.u_total, 3),
                    "e_coulomb": round(f.e_coulomb, 3),
                    "dg_gb_solvation": round(f.dg_gb_solvation, 3),
                    "e_vdw": round(f.e_vdw, 3),
                    "e_hbond": round(f.e_hbond, 3),
                    "e_halogen": round(f.e_halogen, 3),
                    "e_cation_pi": round(f.e_cation_pi, 3),
                    "e_pi_stacking": round(f.e_pi_stacking, 3),
                    "dg_hydrophobic": round(f.dg_hydrophobic, 3),
                    "temp_trans_k": round(f.temp_trans_k, 1),
                    "temp_rot_k": round(f.temp_rot_k, 1),
                    "com_displacement_a": round(f.com_displacement_a, 3),
                    "min_contact_a": round(f.min_contact_a, 3),
                }
                for f in self.frames
            ],
        }


def ligand_masses(params: BiophysicalAtomParams) -> Tensor:
    """Per-atom masses in amu, from atomic numbers via RDKit's periodic table."""
    from rdkit.Chem import GetPeriodicTable

    table = GetPeriodicTable()
    nums = params.atomic_nums.detach().cpu().tolist()
    vals = [float(table.GetAtomicWeight(int(max(1, z)))) for z in nums]
    return torch.tensor(vals, dtype=torch.float32, device=params.coords.device)


def hydrodynamic_radius(coords: Tensor, masses: Tensor) -> float:
    """Equivalent-sphere hydrodynamic radius in Angstrom.

    A uniform sphere of radius ``a`` has radius of gyration ``R_g = a sqrt(3/5)``, so the
    sphere equivalent to a molecule of gyration radius ``R_g`` is ``a = R_g sqrt(5/3)``.
    Deriving the radius from the ligand's own geometry keeps the friction consistent with
    its size instead of pinning it to one hardcoded molecule.
    """
    total = masses.sum()
    com = (coords * masses.unsqueeze(-1)).sum(0) / total
    rg2 = float((masses * ((coords - com) ** 2).sum(-1)).sum() / total)
    return math.sqrt(max(rg2, 1e-6)) * math.sqrt(5.0 / 3.0)


def find_escape_direction(
    lig_coords: Tensor,
    rec_coords: Tensor,
    *,
    n_directions: int = 600,
    max_radius: float = 25.0,
    step: float = 0.5,
) -> Tensor:
    """Find the clearest way out of a pocket, as a unit vector from the ligand centroid.

    The obvious choice -- the vector from the pocket's centroid to the ligand's -- is only
    right for a shallow, open site. For a buried pocket it points at whatever happens to be
    on the far side, which is usually protein. Measured on the D2 dopamine receptor (a 7TM
    GPCR, PDB 6CM4), that naive vector has **0.44 Angstrom of clearance** while the best
    available direction has **3.47 Angstrom and lies 77.5 degrees away**: steering along the
    naive vector drove the ligand into the wall and stalled at a +230 kcal/mol contact
    instead of leaving the pocket.

    This casts rays over a Fibonacci sphere and scores each by its *worst* clearance -- the
    closest any receptor atom comes to the ray anywhere along it -- then returns the
    best-scoring direction. Maximising the minimum is the right objective: a channel is
    only as passable as its narrowest point, so averaging would happily pick a direction
    that is wide open except for one blocking residue.

    Args:
        lig_coords: ``[N, 3]`` ligand coordinates; their centroid is the ray origin.
        rec_coords: ``[M, 3]`` receptor coordinates.
        n_directions: Candidate directions on the sphere. 600 gives roughly 8-degree
            spacing, which is finer than the channels this resolves.
        max_radius: How far along each ray to test, Angstrom.
        step: Sampling interval along a ray, Angstrom.

    Returns:
        A unit vector ``[3]``, on the same device and dtype as ``lig_coords``.
    """
    dev, dtype = lig_coords.device, lig_coords.dtype
    com = lig_coords.mean(0)

    # Fibonacci sphere: near-uniform without the pole clustering of a lat/long grid.
    idx = torch.arange(n_directions, dtype=torch.float64, device=dev) + 0.5
    phi = torch.acos(1.0 - 2.0 * idx / n_directions)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    dirs = torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)],
        dim=1,
    ).to(dtype=dtype)

    # Start at 2 A so the ligand's own neighbourhood does not veto every direction.
    ts = torch.arange(2.0, max_radius, step, dtype=dtype, device=dev)
    # [D, T, 3] sample points -> [D, T] worst clearance per direction.
    pts = com.view(1, 1, 3) + ts.view(1, -1, 1) * dirs.view(-1, 1, 3)
    best_score = torch.full((n_directions,), float("inf"), dtype=dtype, device=dev)
    # Chunk over directions so the [D, T, M] distance tensor never materialises at once.
    chunk = max(1, int(2_000_000 / max(1, ts.numel() * rec_coords.shape[0])))
    for lo in range(0, n_directions, chunk):
        hi = min(n_directions, lo + chunk)
        d = torch.cdist(pts[lo:hi].reshape(-1, 3), rec_coords).min(dim=1).values
        best_score[lo:hi] = d.view(hi - lo, -1).min(dim=1).values
    return dirs[int(torch.argmax(best_score))]


def rigid_body_inertia(coords: Tensor, masses: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(com, principal_moments, principal_axes)``.

    ``I_ab = sum_i m_i (delta_ab |r_i|^2 - r_ia r_ib)`` about the centre of mass, then
    diagonalised so the rotational thermostat can be applied per principal axis -- the
    inertia tensor of a drug-like ligand is strongly anisotropic (often 7x between the
    smallest and largest moment), and a single scalar moment would thermostat the three
    rotational degrees of freedom at three different effective temperatures.

    Args:
        coords: ``[N, 3]`` positions, Angstrom.
        masses: ``[N]`` masses, amu.

    Returns:
        ``com`` ``[3]``; ``moments`` ``[3]`` ascending, amu*A^2; ``axes`` ``[3, 3]`` whose
        *columns* are the principal axes expressed in the lab frame, as a proper rotation.
    """
    total = masses.sum()
    com = (coords * masses.unsqueeze(-1)).sum(0) / total
    rel = coords - com
    r2 = (rel * rel).sum(-1)
    eye = torch.eye(3, dtype=coords.dtype, device=coords.device)
    inertia = (masses * r2).sum() * eye - torch.einsum("i,ia,ib->ab", masses, rel, rel)
    moments, axes = torch.linalg.eigh(0.5 * (inertia + inertia.T))
    # eigh may return an eigenbasis with det = -1, which is a reflection, not a rotation.
    # Feeding a reflection into the quaternion conversion mirrors the ligand -- turning it
    # into its own enantiomer. Flip one axis to restore a right-handed frame.
    if float(torch.det(axes)) < 0:
        axes = axes.clone()
        axes[:, 0] = -axes[:, 0]
    return com, moments.clamp(min=1e-3), axes


def _matrix_to_quat(rot: Tensor) -> Tensor:
    """Unit quaternion ``(w, x, y, z)`` of a proper rotation matrix.

    Uses the largest-diagonal branch rather than the bare ``w = sqrt(1+tr)/2`` formula,
    which loses all precision when the trace approaches -1.
    """
    m = rot
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    m00, m11, m22 = float(m[0, 0]), float(m[1, 1]), float(m[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [
            0.25 * s,
            float(m[2, 1] - m[1, 2]) / s,
            float(m[0, 2] - m[2, 0]) / s,
            float(m[1, 0] - m[0, 1]) / s,
        ]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = [
            float(m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            float(m[0, 1] + m[1, 0]) / s,
            float(m[0, 2] + m[2, 0]) / s,
        ]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = [
            float(m[0, 2] - m[2, 0]) / s,
            float(m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            float(m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = [
            float(m[1, 0] - m[0, 1]) / s,
            float(m[0, 2] + m[2, 0]) / s,
            float(m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]
    out = torch.tensor(q, dtype=rot.dtype, device=rot.device)
    unit: Tensor = out / torch.linalg.vector_norm(out)
    return unit


def _quat_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrix from a unit quaternion ``(w, x, y, z)``."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product of two quaternions ``(w, x, y, z)``."""
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate_about_body_axis(vec: Tensor, axis: int, angle: float) -> Tensor:
    """Rodrigues rotation of ``vec`` about body axis ``axis`` by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    e = torch.zeros(3, dtype=vec.dtype, device=vec.device)
    e[axis] = 1.0
    return c * vec + s * torch.cross(e, vec, dim=-1) + (1.0 - c) * (e @ vec) * e


def _free_top_substep(
    quat: Tensor, ang_mom: Tensor, moments: Tensor, axis: int, tau: float
) -> tuple[Tensor, Tensor]:
    """Exact flow of the single-axis Hamiltonian ``H_a = L_a^2 / (2 I_a)``.

    Under ``H_a`` the component ``L_a`` is conserved, so the body-frame angular momentum
    simply rotates about body axis ``a`` at the constant rate ``L_a / I_a`` while the
    attitude advances by the same angle. Both maps are exact, which is what makes the
    composed scheme conserve ``|L|`` to round-off.
    """
    phi = float(tau * ang_mom[axis] / moments[axis])
    if phi == 0.0:
        return quat, ang_mom
    new_l = _rotate_about_body_axis(ang_mom, axis, -phi)
    e = torch.zeros(3, dtype=quat.dtype, device=quat.device)
    e[axis] = 1.0
    dq = torch.cat(
        [
            torch.full((1,), math.cos(0.5 * phi), dtype=quat.dtype, device=quat.device),
            e * math.sin(0.5 * phi),
        ]
    )
    new_q = _quat_mul(quat, dq)
    return new_q / torch.linalg.vector_norm(new_q), new_l


def _free_top_flow(
    quat: Tensor, ang_mom: Tensor, moments: Tensor, dt: float
) -> tuple[Tensor, Tensor]:
    """Dullweber-Leimkuhler-McLachlan symmetric R-splitting of the torque-free top.

    ``R_1(h/2) R_2(h/2) R_3(h) R_2(h/2) R_1(h/2)`` -- second order by symmetry, and exactly
    angular-momentum conserving because every stage is a rotation.
    """
    quat, ang_mom = _free_top_substep(quat, ang_mom, moments, 0, 0.5 * dt)
    quat, ang_mom = _free_top_substep(quat, ang_mom, moments, 1, 0.5 * dt)
    quat, ang_mom = _free_top_substep(quat, ang_mom, moments, 2, dt)
    quat, ang_mom = _free_top_substep(quat, ang_mom, moments, 1, 0.5 * dt)
    quat, ang_mom = _free_top_substep(quat, ang_mom, moments, 0, 0.5 * dt)
    return quat, ang_mom


def _atom_coords(state: RigidBodyState, body_ref: Tensor) -> Tensor:
    """All-atom lab coordinates from the rigid-body state and body-frame reference."""
    return body_ref @ _quat_to_matrix(state.quat).T + state.com


def simulate_binding(
    lig: BiophysicalAtomParams,
    rec: BiophysicalAtomParams,
    hamiltonian: BiophysicalHamiltonian,
    *,
    config: BindingDynamicsConfig | None = None,
    escape_direction: Tensor | None = None,
    backend: str = "auto",
) -> BindingTrajectory:
    """Run a staged-then-released rigid-body Langevin association trajectory.

    Stage 1 is steered molecular dynamics: a harmonic restraint on the centre of mass whose
    centre travels outward along the escape channel at constant speed, dragging the ligand
    out of the pocket. This is how the launch pose is found, and it has to be dynamics
    rather than geometry: a straight translation out of a buried pocket drives atoms
    through the protein wall (overlaps past 0.5 A, energies past 1e6 kcal/mol), whereas a
    dynamics cannot cross the repulsive wall and instead threads the real channel. Stage 1
    is a staging device and is **not** recorded.

    Stage 2 releases the ligand with fresh Maxwell-Boltzmann velocities and integrates free
    Langevin dynamics. That is the recorded trajectory.

    Args:
        lig: Ligand parameters; the input conformer defines the frozen internal geometry.
        rec: Pocket parameters, held fixed throughout.
        hamiltonian: The force field. Evaluated at ``softcore_lambda=1.0`` only. Prefer
            :func:`make_dynamics_hamiltonian` -- the shipped 10-12 A switching window puts
            a spurious attractive shell right where the approach leg would be animated.
        config: Run parameters.
        escape_direction: Unit vector to steer along. Defaults to
            :func:`find_escape_direction`, which searches for the clearest channel rather
            than assuming the pocket-centroid vector points out of the cavity.
        backend: ``"native"`` runs the C++ engine, ``"python"`` this module's pure-PyTorch
            loop, and ``"auto"`` (the default) prefers native and silently falls back. The
            native engine is roughly 9x faster per force evaluation and agrees with the
            PyTorch potential to about 1e-6 relative -- the float32-versus-float64 noise
            floor -- so ``"python"`` is worth choosing only to cross-check it.

    Returns:
        A :class:`BindingTrajectory` whose ``diagnostics`` carry the physics acceptance
        measurements (effective temperatures, launch displacement, closest contact).

    Raises:
        RuntimeError: If the rigid-body frame fails to reconstruct the input conformer,
            which would mean animating a scrambled or mirrored ligand.
        ValueError: If ``backend`` is not one of the three accepted values.
    """
    if backend not in {"auto", "native", "python"}:
        raise ValueError(f"backend must be auto, native or python, got {backend!r}")
    if backend in {"auto", "native"}:
        try:
            from geneuclidean.biophysics._native_dynamics import simulate_binding_native

            return simulate_binding_native(
                lig,
                rec,
                hamiltonian,
                config=config,
                escape_direction=(
                    None if escape_direction is None else escape_direction.detach().cpu().numpy()
                ),
            )
        except Exception:
            if backend == "native":
                raise

    cfg = config or BindingDynamicsConfig()
    dev = lig.coords.device
    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))

    masses = ligand_masses(lig)
    total_mass = float(masses.sum())
    com0, moments, axes = rigid_body_inertia(lig.coords.detach(), masses)
    # Body-frame reference geometry, expressed in the principal-axis basis so the inertia
    # tensor is diagonal and each rotational degree of freedom can be thermostatted.
    body_ref = (lig.coords.detach() - com0) @ axes
    rec_coords = rec.coords.detach()

    radius = (
        float(cfg.hydrodynamic_radius_a)
        if cfg.hydrodynamic_radius_a is not None
        else hydrodynamic_radius(lig.coords.detach(), masses)
    )
    zeta_t = SIX_PI_ETA_WATER_300K * radius
    gamma_t = float(cfg.gamma_trans) if cfg.gamma_trans is not None else zeta_t / total_mass
    zeta_r = (
        float(cfg.zeta_rot) if cfg.zeta_rot is not None else EIGHT_PI_ETA_WATER_300K * radius**3
    )
    rot_rates = zeta_r / moments  # [3] per-axis Ornstein-Uhlenbeck rates, 1/ps

    if escape_direction is None:
        # Not the pocket-centroid vector: for a buried site that aims at the far wall.
        escape_direction = find_escape_direction(lig.coords.detach(), rec_coords)
    escape = (escape_direction / torch.linalg.vector_norm(escape_direction)).to(dev)

    kt = cfg.temperature_k * KB_KCAL_MOL_K
    kt_red = MD_ACCEL_FACTOR * kt  # amu*A^2/ps^2
    dt = float(cfg.timestep_ps)

    # The starting orientation is `axes`, NOT the identity: `body_ref` lives in the
    # principal-axis frame, so the rotation that maps it back onto the input conformer is
    # exactly the eigenbasis. Starting from the identity quaternion silently re-orients the
    # ligand inside the pocket and produces a ~1e7 kcal/mol/A clash on step one.
    state = RigidBodyState(
        com=com0.clone(),
        quat=_matrix_to_quat(axes).to(device=dev, dtype=torch.float32),
        vel=torch.zeros(3, dtype=torch.float32, device=dev),
        ang_mom=torch.zeros(3, dtype=torch.float32, device=dev),
    )
    # Fail loudly rather than animate a scrambled ligand: the reconstructed pose must
    # reproduce the input conformer to within float32 round-off.
    rebuild_err = float((_atom_coords(state, body_ref) - lig.coords.detach()).abs().max())
    if rebuild_err > 1e-3:
        raise RuntimeError(
            f"rigid-body frame does not reconstruct the input conformer "
            f"(max coordinate error {rebuild_err:.3e} A); refusing to integrate"
        )

    c1t = math.exp(-gamma_t * dt)
    c2t = math.sqrt(max(0.0, 1.0 - c1t * c1t))
    c1r = torch.exp(-rot_rates * dt)
    c2r = torch.sqrt((1.0 - c1r * c1r).clamp(min=0.0))
    # Stationary std of each body-frame angular-momentum component is sqrt(I_a kT): the
    # noise strength sqrt(2 zeta_r kT) is axis-independent, the rate zeta_r/I_a is not.
    l_sigma = torch.sqrt(moments * kt_red)

    def net_force_and_torque(coords: Tensor) -> tuple[Tensor, Tensor]:
        """Net force (kcal/mol/A) and lab-frame torque. Raw autograd, no clipping."""
        c = coords.detach().clone().requires_grad_(True)
        u = hamiltonian.potential_energy(c, rec_coords, lig, rec, softcore_lambda=1.0)
        (grad,) = torch.autograd.grad(u, c)
        f_atoms = -grad.detach()
        rel = (coords.detach() - state.com).detach()
        return f_atoms.sum(0), torch.cross(rel, f_atoms, dim=-1).sum(0)

    def baoab_step(extra_force: Tensor | None, *, thermostat: bool) -> float:
        """One BAOAB step of the six rigid-body degrees of freedom.

        Returns the magnitude of the applied external (steering) force, for diagnostics.
        """
        net, tq_lab = net_force_and_torque(_atom_coords(state, body_ref))
        if extra_force is not None:
            net = net + extra_force
        tq_b = _quat_to_matrix(state.quat).T @ tq_lab

        # B: half kick. No gyroscopic term -- the free-top flow in A handles it exactly.
        state.vel = state.vel + 0.5 * dt * MD_ACCEL_FACTOR * net / total_mass
        state.ang_mom = state.ang_mom + 0.5 * dt * MD_ACCEL_FACTOR * tq_b

        # A: half drift, translation plus exact torque-free rotation
        state.com = state.com + 0.5 * dt * state.vel
        state.quat, state.ang_mom = _free_top_flow(state.quat, state.ang_mom, moments, 0.5 * dt)

        # O: exact Ornstein-Uhlenbeck on the COM velocity and the body-frame angular
        # momentum, the latter with three distinct rates zeta_r / I_a.
        if thermostat:
            xi = torch.randn(3, generator=gen).to(dev)
            state.vel = c1t * state.vel + c2t * math.sqrt(kt_red / total_mass) * xi
            eta = torch.randn(3, generator=gen).to(dev)
            state.ang_mom = c1r * state.ang_mom + c2r * l_sigma * eta

        # A: half drift
        state.com = state.com + 0.5 * dt * state.vel
        state.quat, state.ang_mom = _free_top_flow(state.quat, state.ang_mom, moments, 0.5 * dt)

        # B: half kick at the new configuration
        net2, tq_lab2 = net_force_and_torque(_atom_coords(state, body_ref))
        if extra_force is not None:
            net2 = net2 + extra_force
        tq_b2 = _quat_to_matrix(state.quat).T @ tq_lab2
        state.vel = state.vel + 0.5 * dt * MD_ACCEL_FACTOR * net2 / total_mass
        state.ang_mom = state.ang_mom + 0.5 * dt * MD_ACCEL_FACTOR * tq_b2
        return 0.0 if extra_force is None else float(torch.linalg.vector_norm(extra_force))

    # ---- Stage 1: steered MD outward to find a clash-free launch pose (not recorded) ----
    staged = 0
    max_pull = 0.0
    for step in range(int(cfg.stage_max_steps)):
        # Restraint centre marches outward at constant speed, capped at the target.
        travelled = min(cfg.stage_pull_velocity * step * dt, cfg.launch_distance_a)
        target = com0 + escape * travelled
        pull = cfg.stage_spring_k * (target - state.com)
        max_pull = max(max_pull, baoab_step(pull, thermostat=True))
        staged += 1
        if float(torch.linalg.vector_norm(state.com - com0)) >= cfg.launch_distance_a:
            break
    launch_displacement = float(torch.linalg.vector_norm(state.com - com0))

    # ---- Stage 2: release and record ----
    # Fresh Maxwell-Boltzmann draw so the recorded run does not inherit steering momentum.
    state.vel = math.sqrt(kt_red / total_mass) * torch.randn(3, generator=gen).to(dev)
    state.ang_mom = l_sigma * torch.randn(3, generator=gen).to(dev)

    traj = BindingTrajectory()
    ke_t_acc, ke_r_acc, n_acc = 0.0, 0.0, 0
    closest = float("inf")

    for step in range(int(cfg.release_steps)):
        baoab_step(None, thermostat=True)

        ke_t = 0.5 * total_mass * float((state.vel * state.vel).sum()) / MD_ACCEL_FACTOR
        ke_r = 0.5 * float((state.ang_mom * state.ang_mom / moments).sum()) / MD_ACCEL_FACTOR
        ke_t_acc += ke_t
        ke_r_acc += ke_r
        n_acc += 1

        if step % int(max(1, cfg.record_every)) == 0 or step == int(cfg.release_steps) - 1:
            coords = _atom_coords(state, body_ref).detach()
            bd = hamiltonian.evaluate_complex(lig, rec, lig_coords=coords, rec_coords=rec_coords)
            with torch.no_grad():
                contact = float((coords.unsqueeze(1) - rec_coords.unsqueeze(0)).norm(dim=-1).min())
            closest = min(closest, contact)
            traj.frames.append(
                TrajectoryFrame(
                    step=step,
                    time_ps=step * dt,
                    com=[float(v) for v in state.com],
                    quat=[float(v) for v in state.quat],
                    coords=[[float(c) for c in xyz] for xyz in coords],
                    u_total=float(bd.u_total),
                    e_coulomb=float(bd.e_coulomb),
                    dg_gb_solvation=float(bd.dg_gb_solvation),
                    e_vdw=float(bd.e_vdw),
                    e_hbond=float(bd.e_hbond),
                    e_halogen=float(bd.e_halogen),
                    e_cation_pi=float(bd.e_cation_pi),
                    e_pi_stacking=float(bd.e_pi_stacking),
                    dg_hydrophobic=float(bd.dg_hydrophobic),
                    temp_trans_k=2.0 * ke_t / (3.0 * KB_KCAL_MOL_K),
                    temp_rot_k=2.0 * ke_r / (3.0 * KB_KCAL_MOL_K),
                    com_displacement_a=float(torch.linalg.vector_norm(state.com - com0)),
                    min_contact_a=contact,
                )
            )

    # Equipartition is the headline acceptance test: each of the three translational and
    # three rotational degrees of freedom should hold (1/2)kT on average, so each group of
    # three holds (3/2)kT = 0.894242 kcal/mol at 300 K.
    mean_ke_t = ke_t_acc / max(1, n_acc)
    mean_ke_r = ke_r_acc / max(1, n_acc)
    traj.diagnostics = {
        "backend": "python (PyTorch autograd, float32)",
        "n_frames": len(traj.frames),
        "total_mass_amu": round(total_mass, 3),
        "principal_moments_amu_a2": [round(float(v), 2) for v in moments],
        "hydrodynamic_radius_a": round(radius, 4),
        "gamma_trans_per_ps": round(gamma_t, 4),
        "zeta_rot_amu_a2_per_ps": round(zeta_r, 1),
        "rot_rates_per_ps": [round(float(v), 2) for v in rot_rates],
        "rot_rate_times_dt": [round(float(v) * dt, 4) for v in rot_rates],
        "timestep_ps": dt,
        "release_steps": int(cfg.release_steps),
        "simulated_time_ps": round(int(cfg.release_steps) * dt, 4),
        "temperature_target_k": cfg.temperature_k,
        "stage_steps_used": staged,
        "stage_max_pull_force": round(max_pull, 2),
        "launch_displacement_a": round(launch_displacement, 3),
        "mean_ke_trans_kcal_mol": round(mean_ke_t, 5),
        "mean_ke_rot_kcal_mol": round(mean_ke_r, 5),
        "expected_ke_per_3dof_kcal_mol": round(1.5 * kt, 5),
        "temp_trans_effective_k": round(2.0 * mean_ke_t / (3.0 * KB_KCAL_MOL_K), 1),
        "temp_rot_effective_k": round(2.0 * mean_ke_r / (3.0 * KB_KCAL_MOL_K), 1),
        "final_com_displacement_a": round(float(torch.linalg.vector_norm(state.com - com0)), 3),
        "closest_contact_a": round(closest, 3),
        "frame_rebuild_error_a": float(f"{rebuild_err:.3e}"),
        "softcore_lambda": 1.0,
        "force_clipping": "none (raw autograd gradient)",
        "rotational_integrator": "DLM symmetric free-top R-splitting (2nd order)",
        "switch_on_a": float(getattr(hamiltonian, "switch_on", float("nan"))),
        "nonbonded_cutoff_a": float(getattr(hamiltonian, "nonbonded_cutoff", float("nan"))),
        "contract": (
            "Boltzmann sampling of a fixed empirical scoring function at a nominal "
            "temperature; rigid ligand, rigid receptor. Not solvated binding kinetics."
        ),
    }
    return traj
