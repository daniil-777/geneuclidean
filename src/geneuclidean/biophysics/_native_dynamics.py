"""ctypes bridge to the native rigid-body binding-dynamics engine.

The PyTorch Hamiltonian costs 77.6 ms per energy+gradient call on a 40-atom ligand in a
962-atom pocket, almost all of it kernel-launch overhead rather than arithmetic (see
``_native/binding_dynamics.cpp`` for the measurements). This module compiles that engine
and feeds it the constants that a frozen receptor and a rigid ligand make constant.

Correctness discipline
----------------------
Every cached constant is produced by the **existing verified Python implementation**
wherever one exists -- the apo SASA from ``_compute_sasa_exposed``, the donor axes from
``compute_donor_axes_from_coords``, the ring frames from
``compute_ring_geometry_from_coords``. Only the HCT descreening *integral* has to be
mirrored in NumPy, because ``compute_effective_born_radii`` returns the closed radius and
never exposes the integral. That mirror is not taken on trust: :meth:`NativeEngine.build`
closes its own integrals through the same OBC2 formula and checks the result against
``compute_effective_born_radii``, raising if they disagree. A silent divergence there would
be invisible in the animation and fatal to the physics.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geneuclidean.biophysics.binding_dynamics import (
        BindingDynamicsConfig,
        BindingTrajectory,
    )
    from geneuclidean.biophysics.forcefield import (
        BiophysicalAtomParams,
        BiophysicalHamiltonian,
    )

__all__ = [
    "NativeEngine",
    "build_native_library",
    "build_wasm_module",
    "native_available",
    "pack_engine_blob",
    "simulate_binding_native",
]

_SRC = Path(__file__).parent / "_native" / "binding_dynamics.cpp"
_ABI_VERSION = 7
#: Doubles per frame record before the coordinates (see gex_simulate's docstring).
_FRAME_HEADER = 21


def _default_lib_path() -> Path:
    env = os.environ.get("GENEUCLIDEAN_NATIVE_LIB")
    if env:
        return Path(env)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return _SRC.parent / f"libgexdyn{suffix}"


def build_native_library(out_path: Path | None = None, *, force: bool = False) -> Path:
    """Compile the native engine, reusing an existing build newer than the source.

    The engine deliberately uses ``std::thread`` rather than OpenMP. PyTorch already has an
    OpenMP runtime resident in the process, and loading a second ``libomp`` through ctypes
    crashes inside ``__kmp_suspend_initialize_thread`` on macOS. Threading by hand also puts
    the parallelism at a better granularity -- across the twelve finite-difference energy
    evaluations of a force call rather than inside a 38k-pair loop.

    Args:
        out_path: Where to write the shared library. Defaults to beside the C++ source, or
            to ``$GENEUCLIDEAN_NATIVE_LIB`` when that is set.
        force: Rebuild even if the library looks current.

    Returns:
        Path to the compiled shared library.

    Raises:
        FileNotFoundError: If the C++ source is missing.
        RuntimeError: If compilation fails; the compiler's own output is included.
    """
    if not _SRC.is_file():
        raise FileNotFoundError(f"native source missing: {_SRC}")
    out = Path(out_path) if out_path is not None else _default_lib_path()
    if out.is_file() and not force and out.stat().st_mtime >= _SRC.stat().st_mtime:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    cxx = os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++")
    cmd = [cxx, "-std=c++17", "-O3", "-fPIC", "-shared"]
    if sys.platform == "darwin":
        cmd.append("-march=native")
    # -ffp-contract=off keeps the arithmetic reproducible across builds and machines.
    cmd += ["-ffp-contract=off", "-fvisibility=hidden", "-pthread"]
    cmd += [str(_SRC), "-o", str(out)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            "failed to compile the native dynamics engine\n"
            f"command: {' '.join(cmd)}\n{proc.stderr or proc.stdout}"
        )
    return out


class _GexParams(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_double)
        for name in (
            "interior_eps",
            "solvent_eps",
            "gb_weight",
            "switch_on",
            "cutoff",
            "hbond_well",
            "salt_boost",
            "hal_crown",
            "hal_belt",
            "cation_pi",
            "pi_pd",
            "pi_t",
            "sasa_tension",
            "r_max_born",
            "use_gb",
            "gb_changed_tol",
        )
    ]


_D = ctypes.POINTER(ctypes.c_double)
_U8 = ctypes.POINTER(ctypes.c_uint8)


def _load(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    lib.gex_version.restype = ctypes.c_int
    lib.gex_version.argtypes = []
    got = int(lib.gex_version())
    if got != _ABI_VERSION:
        raise RuntimeError(
            f"native library ABI {got} does not match the expected {_ABI_VERSION}; "
            "rebuild with build_native_library(force=True)"
        )
    lib.gex_create.restype = ctypes.c_void_p
    # Annotated because ctypes types the attribute as a Sequence of its own union, which a
    # list built by concatenation does not join cleanly.
    create_argtypes: list[Any] = (
        [ctypes.c_int] * 4
        + [_D] * 5
        + [_D, _U8]
        + [_D] * 4
        + [_D, _U8]
        + [_D] * 9
        + [ctypes.POINTER(_GexParams)]
    )
    lib.gex_create.argtypes = create_argtypes
    lib.gex_destroy.restype = None
    lib.gex_destroy.argtypes = [ctypes.c_void_p]
    lib.gex_energy.restype = ctypes.c_double
    lib.gex_energy.argtypes = [ctypes.c_void_p, _D, _D, _D]
    lib.gex_forces.restype = None
    lib.gex_forces.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_double, ctypes.c_double, _D, _D]
    lib.gex_pose.restype = None
    lib.gex_pose.argtypes = [ctypes.c_void_p, _D, _D, _D]
    lib.gex_simulate.restype = ctypes.c_int
    lib.gex_simulate.argtypes = [
        ctypes.c_void_p,
        _D,
        _D,
        _D,
        _D,
        _D,
        ctypes.c_double,
        _D,
        ctypes.c_int,
        _D,
    ]
    return lib


def native_available() -> bool:
    """True if the native engine can be compiled and loaded on this machine."""
    try:
        _load(build_native_library())
        return True
    except Exception:
        return False


def _p(a: np.ndarray) -> Any:
    """Pointer to a contiguous float64 array."""
    return a.ctypes.data_as(_D)


def _pu8(a: np.ndarray) -> Any:
    return a.ctypes.data_as(_U8)


# ---------------------------------------------------------------------------------------
# NumPy mirrors of the two formulas the Python API does not expose directly.
# ---------------------------------------------------------------------------------------


def _hct_integral(coords: np.ndarray, rho: np.ndarray, obc: np.ndarray) -> np.ndarray:
    """Intra-molecular HCT descreening integral per atom, mirroring forcefield.py.

    Returned **unclamped**, because the reference clamps the complete complex sum, not the
    intra part -- clamping here would make the split ``I = I_intra + I_cross`` inexact.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    rho_c = np.maximum(rho, 0.15)
    rho_t = np.maximum(rho_c - 0.09, 0.10)
    scaled = obc * rho_t

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1) + 1e-12)
    eye = np.eye(n, dtype=bool)
    dist = np.where(eye, 1.0, dist)

    rho_i = rho_t[:, None]
    sr_j = scaled[None, :]
    upper = dist + sr_j
    lower = np.maximum(rho_i, np.abs(dist - sr_j))
    inv_l = 1.0 / np.maximum(lower, 1e-6)
    inv_u = 1.0 / np.maximum(upper, 1e-6)

    bracket = (
        inv_l
        - inv_u
        + 0.25 * (dist - sr_j * sr_j / dist) * (inv_u * inv_u - inv_l * inv_l)
        + 0.5 * np.log(lower / upper) / dist
    )
    engulfed = rho_i < (sr_j - dist)
    term = 0.5 * (bracket + np.where(engulfed, 2.0 * (1.0 / rho_i - inv_l), 0.0))
    term = np.where(rho_i < upper, term, 0.0)
    term = np.where(eye, 0.0, term)
    return np.asarray(term.sum(-1), dtype=np.float64)


def _obc2_close(integral: np.ndarray, rho: np.ndarray, r_max: float = 30.0) -> np.ndarray:
    """OBC2 closing of a descreening integral into an effective Born radius."""
    rho_c = np.maximum(rho, 0.15)
    rho_t = np.maximum(rho_c - 0.09, 0.10)
    psi = np.maximum(integral, 0.0) * rho_t
    arg = psi * (1.0 - 0.8 * psi + 4.85 * psi * psi)
    inv = np.maximum((1.0 / rho_t) - np.tanh(arg) / rho_c, 1.0 / r_max)
    return np.asarray(1.0 / inv, dtype=np.float64)


def _still_f_gb(d2: np.ndarray, ri: np.ndarray, rj: np.ndarray) -> np.ndarray:
    ri_rj = ri[:, None] * rj[None, :]
    return np.asarray(
        np.sqrt(d2 + ri_rj * np.exp(-d2 / (4.0 * ri_rj + 1e-8)) + 1e-8), dtype=np.float64
    )


@dataclass
class NativeEngine:
    """A built native engine bound to one ligand/pocket pair and one Hamiltonian."""

    handle: int
    lib: ctypes.CDLL
    n_lig: int
    n_rec: int
    com0: np.ndarray
    quat0: np.ndarray
    moments: np.ndarray
    axes: np.ndarray
    masses: np.ndarray
    body_ref: np.ndarray
    born_check: float
    # The exact buffers handed to gex_create, kept so pack_engine_blob can serialise the
    # identical engine for the WebAssembly build instead of recomputing it.
    _n_lring: int = 0
    _n_rring: int = 0
    _params: Any = None
    _packed_arrays: tuple = ()
    _packed_flags: tuple = ()

    # -- lifecycle -----------------------------------------------------------------
    def close(self) -> None:
        """Release the native engine. Safe to call more than once."""
        if self.handle:
            self.lib.gex_destroy(ctypes.c_void_p(self.handle))
            self.handle = 0

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        with contextlib.suppress(Exception):
            self.close()

    # -- queries -------------------------------------------------------------------
    def energy(self, quat: np.ndarray, com: np.ndarray) -> tuple[float, np.ndarray]:
        """Total energy and the eight-term breakdown at a pose."""
        q = np.ascontiguousarray(quat, dtype=np.float64)
        c = np.ascontiguousarray(com, dtype=np.float64)
        terms = np.zeros(8, dtype=np.float64)
        total = float(self.lib.gex_energy(ctypes.c_void_p(self.handle), _p(q), _p(c), _p(terms)))
        return total, terms

    def forces(
        self, quat: np.ndarray, com: np.ndarray, *, h_trans: float = 1e-5, h_rot: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Net force (kcal/mol/A) and lab-frame torque (kcal/mol) by central differences."""
        q = np.ascontiguousarray(quat, dtype=np.float64)
        c = np.ascontiguousarray(com, dtype=np.float64)
        f = np.zeros(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        self.lib.gex_forces(
            ctypes.c_void_p(self.handle), _p(q), _p(c), h_trans, h_rot, _p(f), _p(t)
        )
        return f, t

    def pose(self, quat: np.ndarray, com: np.ndarray) -> np.ndarray:
        """Lab-frame ligand coordinates ``[n_lig, 3]`` at a pose."""
        q = np.ascontiguousarray(quat, dtype=np.float64)
        c = np.ascontiguousarray(com, dtype=np.float64)
        out = np.zeros(3 * self.n_lig, dtype=np.float64)
        self.lib.gex_pose(ctypes.c_void_p(self.handle), _p(q), _p(c), _p(out))
        return out.reshape(self.n_lig, 3)

    # -- construction --------------------------------------------------------------
    @staticmethod
    def build(
        lig: BiophysicalAtomParams,
        rec: BiophysicalAtomParams,
        hamiltonian: BiophysicalHamiltonian,
        *,
        lib_path: Path | None = None,
        born_tolerance: float = 2e-4,
        frame_tolerance: float = 1e-4,
        gb_changed_tol: float = 0.0,
    ) -> NativeEngine:
        """Build a native engine, validating the mirrored Born-radius formula on the way.

        Raises:
            RuntimeError: If the NumPy descreening integral, closed through OBC2, fails to
                reproduce ``compute_effective_born_radii`` within ``born_tolerance``, or if
                the rigid-body frame fails to reconstruct the input conformer.
        """
        from geneuclidean.biophysics.binding_dynamics import (
            _matrix_to_quat,
            ligand_masses,
            rigid_body_inertia,
        )
        from geneuclidean.biophysics.forcefield import (
            compute_donor_axes_from_coords,
            compute_halogen_axes_from_coords,
            compute_ring_geometry_from_coords,
        )

        lib = _load(build_native_library(lib_path))

        lc_t = lig.coords.detach()
        rc_t = rec.coords.detach()
        nl, nrec = int(lc_t.shape[0]), int(rc_t.shape[0])

        masses = ligand_masses(lig)
        com0_t, moments_t, axes_t = rigid_body_inertia(lc_t, masses)
        body_ref_t = (lc_t - com0_t) @ axes_t

        lc = lc_t.numpy().astype(np.float64)
        rc = np.ascontiguousarray(rc_t.numpy().astype(np.float64))
        axes = axes_t.numpy().astype(np.float64)
        com0 = com0_t.numpy().astype(np.float64)
        body_ref = np.ascontiguousarray(body_ref_t.numpy().astype(np.float64))

        def f64(t: Any) -> np.ndarray:
            return np.asarray(t.detach().numpy(), dtype=np.float64)

        def u8(t: Any) -> np.ndarray:
            return np.asarray(t.detach().numpy(), dtype=np.uint8)

        # --- ligand frames, expressed in the body (principal-axis) basis ---
        d_ax = f64(
            compute_donor_axes_from_coords(
                lc_t, lig.is_donor, lig.donor_h_pairs, lig.bond_pairs, lig.donor_axes
            )
        )
        h_ax = f64(
            compute_halogen_axes_from_coords(lc_t, lig.halogen_bond_pairs, lig.halogen_sigma_axes)
        )
        lig_daxes_body = np.ascontiguousarray(d_ax @ axes)
        lig_haxes_body = np.ascontiguousarray(h_ax @ axes)

        lrc_t, lrn_t = compute_ring_geometry_from_coords(
            lc_t, lig.ring_atom_indices, lig.ring_atom_mask
        )
        n_lring = int(lrc_t.shape[0])
        if n_lring:
            lig_ringc_body = np.ascontiguousarray((f64(lrc_t) - com0) @ axes)
            lig_ringn_body = np.ascontiguousarray(f64(lrn_t) @ axes)
        else:
            lig_ringc_body = np.zeros((1, 3), dtype=np.float64)
            lig_ringn_body = np.zeros((1, 3), dtype=np.float64)

        # --- receptor frames, constant in the lab ---
        rec_daxes = np.ascontiguousarray(
            f64(
                compute_donor_axes_from_coords(
                    rc_t, rec.is_donor, rec.donor_h_pairs, rec.bond_pairs, rec.donor_axes
                )
            )
        )
        rrc_t, rrn_t = compute_ring_geometry_from_coords(
            rc_t, rec.ring_atom_indices, rec.ring_atom_mask
        )
        n_rring = int(rrc_t.shape[0])
        rec_ringc = np.ascontiguousarray(
            f64(rrc_t) if n_rring else np.zeros((1, 3), dtype=np.float64)
        )
        rec_ringn = np.ascontiguousarray(
            f64(rrn_t) if n_rring else np.zeros((1, 3), dtype=np.float64)
        )

        z_lig = lig.atomic_nums.detach().numpy()
        r0x = np.where(z_lig == 53, 3.45, np.where(z_lig == 35, 3.30, 3.20)).astype(np.float64)

        lig_scalar = np.ascontiguousarray(
            np.concatenate(
                [
                    f64(lig.partial_charges),
                    f64(lig.vdw_sigma),
                    f64(lig.vdw_epsilon),
                    f64(lig.born_radii),
                    f64(lig.obc_scaling),
                    f64(lig.halogen_strength),
                    r0x,
                ]
            )
        )
        lig_flags = np.ascontiguousarray(
            np.concatenate(
                [
                    u8(lig.is_donor),
                    u8(lig.is_acceptor),
                    u8(lig.is_cationic),
                    u8(lig.is_anionic),
                    u8(lig.is_hydrophobic),
                    u8(lig.is_halogen),
                ]
            )
        )
        rec_scalar = np.ascontiguousarray(
            np.concatenate(
                [
                    f64(rec.partial_charges),
                    f64(rec.vdw_sigma),
                    f64(rec.vdw_epsilon),
                    f64(rec.born_radii),
                    f64(rec.obc_scaling),
                ]
            )
        )
        rec_flags = np.ascontiguousarray(
            np.concatenate(
                [
                    u8(rec.is_donor),
                    u8(rec.is_acceptor),
                    u8(rec.is_cationic),
                    u8(rec.is_anionic),
                    u8(rec.is_hydrophobic),
                ]
            )
        )

        # --- the caches that rigidity makes valid ---
        lig_rho = f64(lig.born_radii)
        rec_rho = f64(rec.born_radii)
        lig_intra = _hct_integral(lc, lig_rho, f64(lig.obc_scaling))
        rec_intra = _hct_integral(rc, rec_rho, f64(rec.obc_scaling))
        lig_born_apo = _obc2_close(lig_intra, lig_rho)
        rec_born_apo = _obc2_close(rec_intra, rec_rho)

        # Validate the mirror: close our own integral and compare against the reference.
        ref_l = f64(hamiltonian.compute_effective_born_radii(lc_t, lig.born_radii, lig.obc_scaling))
        ref_r = f64(hamiltonian.compute_effective_born_radii(rc_t, rec.born_radii, rec.obc_scaling))
        err_l = float(np.max(np.abs(lig_born_apo - ref_l))) if nl else 0.0
        err_r = float(np.max(np.abs(rec_born_apo - ref_r))) if nrec else 0.0
        born_check = max(err_l, err_r)
        if born_check > born_tolerance:
            raise RuntimeError(
                "the NumPy HCT descreening mirror disagrees with "
                f"compute_effective_born_radii by {born_check:.3e} A "
                f"(ligand {err_l:.3e}, receptor {err_r:.3e}); refusing to run so the "
                "native engine cannot silently sample a different potential"
            )

        sasa_l = f64(hamiltonian._compute_sasa_exposed(lc_t, lig.vdw_sigma, lig.is_hydrophobic))
        sasa_r = f64(hamiltonian._compute_sasa_exposed(rc_t, rec.vdw_sigma, rec.is_hydrophobic))

        lig_d2 = np.ascontiguousarray(((lc[:, None, :] - lc[None, :, :]) ** 2).sum(-1))
        rec_d2 = np.ascontiguousarray(((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1))
        rec_f_apo = np.ascontiguousarray(_still_f_gb(rec_d2, rec_born_apo, rec_born_apo))

        params = _GexParams(
            interior_eps=float(hamiltonian.interior_dielectric),
            solvent_eps=float(hamiltonian.solvent_dielectric),
            gb_weight=float(hamiltonian.gb_weight),
            switch_on=float(hamiltonian.switch_on),
            cutoff=float(hamiltonian.nonbonded_cutoff),
            hbond_well=float(hamiltonian.hbond_well_kcal),
            salt_boost=float(hamiltonian.salt_bridge_boost),
            hal_crown=float(hamiltonian.halogen_crown_kcal),
            hal_belt=float(hamiltonian.halogen_belt_kcal),
            cation_pi=float(hamiltonian.cation_pi_kcal),
            pi_pd=float(hamiltonian.pi_pd_kcal),
            pi_t=float(hamiltonian.pi_t_kcal),
            sasa_tension=float(hamiltonian.sasa_surface_tension),
            r_max_born=30.0,
            use_gb=1.0 if hamiltonian.use_gb_solvation else 0.0,
            gb_changed_tol=gb_changed_tol,
        )

        # gex_create copies everything, but the pointers must stay valid for the call.
        bufs = [
            body_ref,
            lig_daxes_body,
            lig_haxes_body,
            lig_ringc_body,
            lig_ringn_body,
            lig_scalar,
            lig_flags,
            rc,
            rec_daxes,
            rec_ringc,
            rec_ringn,
            rec_scalar,
            rec_flags,
            np.ascontiguousarray(lig_intra),
            np.ascontiguousarray(rec_intra),
            np.ascontiguousarray(lig_born_apo),
            np.ascontiguousarray(rec_born_apo),
            np.ascontiguousarray(sasa_l),
            np.ascontiguousarray(sasa_r),
            lig_d2,
            rec_d2,
            rec_f_apo,
        ]
        handle = lib.gex_create(
            nl,
            nrec,
            n_lring,
            n_rring,
            _p(bufs[0]),
            _p(bufs[1]),
            _p(bufs[2]),
            _p(bufs[3]),
            _p(bufs[4]),
            _p(bufs[5]),
            _pu8(bufs[6]),
            _p(bufs[7]),
            _p(bufs[8]),
            _p(bufs[9]),
            _p(bufs[10]),
            _p(bufs[11]),
            _pu8(bufs[12]),
            _p(bufs[13]),
            _p(bufs[14]),
            _p(bufs[15]),
            _p(bufs[16]),
            _p(bufs[17]),
            _p(bufs[18]),
            _p(bufs[19]),
            _p(bufs[20]),
            _p(bufs[21]),
            ctypes.byref(params),
        )
        if not handle:
            raise RuntimeError("gex_create returned null")

        # Starting attitude is the principal-axis eigenbasis, not the identity.
        quat0 = f64(_matrix_to_quat(axes_t))

        eng = NativeEngine(
            handle=int(handle),
            lib=lib,
            n_lig=nl,
            n_rec=nrec,
            com0=com0,
            quat0=quat0,
            moments=f64(moments_t),
            axes=axes,
            masses=f64(masses),
            body_ref=body_ref,
            born_check=born_check,
            _n_lring=n_lring,
            _n_rring=n_rring,
            _params=params,
            # Same order as gex_create's parameter list, which is also the blob's order.
            _packed_arrays=(
                bufs[0],
                bufs[1],
                bufs[2],
                bufs[3],
                bufs[4],
                bufs[5],
                bufs[7],
                bufs[8],
                bufs[9],
                bufs[10],
                bufs[11],
                bufs[13],
                bufs[14],
                bufs[15],
                bufs[16],
                bufs[17],
                bufs[18],
                bufs[19],
                bufs[20],
                bufs[21],
            ),
            _packed_flags=(bufs[6], bufs[12]),
        )
        # Same guard as the Python path: refuse to animate a scrambled or mirrored ligand.
        # The tolerance cannot be tighter than the inputs: coordinates arrive as float32,
        # so at a 50 A extent they already carry ~6e-6 A of absolute round-off, which the
        # centre-of-mass subtraction and the eigenbasis round-trip then compound. 1e-4 A
        # sits two orders above that noise and three below any displacement that would
        # change an energy.
        err = float(np.max(np.abs(eng.pose(quat0, com0) - lc)))
        if err > frame_tolerance:
            eng.close()
            raise RuntimeError(
                "native rigid-body frame does not reconstruct the input conformer "
                f"(max coordinate error {err:.3e} A)"
            )
        return eng


def simulate_binding_native(
    lig: BiophysicalAtomParams,
    rec: BiophysicalAtomParams,
    hamiltonian: BiophysicalHamiltonian,
    *,
    config: BindingDynamicsConfig | None = None,
    escape_direction: np.ndarray | None = None,
    h_trans: float = 1e-5,
    h_rot: float = 1e-5,
    engine: NativeEngine | None = None,
) -> BindingTrajectory:
    """Native equivalent of ``binding_dynamics.simulate_binding``.

    Returns the same :class:`BindingTrajectory` dataclass, so callers and the viewer cannot
    tell the two backends apart except by the ``backend`` diagnostic.

    Raises:
        RuntimeError: If the preallocated frame buffer is too small for the requested run.
    """
    import torch

    from geneuclidean.biophysics.binding_dynamics import (
        EIGHT_PI_ETA_WATER_300K,
        SIX_PI_ETA_WATER_300K,
        BindingDynamicsConfig,
        BindingTrajectory,
        TrajectoryFrame,
        hydrodynamic_radius,
    )
    from geneuclidean.biophysics.forcefield import KB_KCAL_MOL_K

    cfg = config or BindingDynamicsConfig()
    eng = engine if engine is not None else NativeEngine.build(lig, rec, hamiltonian)
    try:
        radius = (
            float(cfg.hydrodynamic_radius_a)
            if cfg.hydrodynamic_radius_a is not None
            else hydrodynamic_radius(
                lig.coords.detach(), torch.from_numpy(eng.masses.astype(np.float32))
            )
        )
        total_mass = float(eng.masses.sum())
        gamma_t = (
            float(cfg.gamma_trans)
            if cfg.gamma_trans is not None
            else SIX_PI_ETA_WATER_300K * radius / total_mass
        )
        zeta_r = (
            float(cfg.zeta_rot) if cfg.zeta_rot is not None else EIGHT_PI_ETA_WATER_300K * radius**3
        )

        if escape_direction is not None:
            esc = np.asarray(escape_direction, dtype=np.float64).reshape(3)
        else:
            # Same channel search as the Python backend; the pocket-centroid vector aims
            # at the far wall for a buried site.
            from geneuclidean.biophysics.binding_dynamics import find_escape_direction

            esc = (
                find_escape_direction(lig.coords.detach(), rec.coords.detach())
                .numpy()
                .astype(np.float64)
            )
        esc = np.ascontiguousarray(esc / max(float(np.linalg.norm(esc)), 1e-6))

        n_expected = int(cfg.release_steps) // max(1, int(cfg.record_every)) + 2
        stride = _FRAME_HEADER + 3 * eng.n_lig
        frames_buf = np.zeros(n_expected * stride, dtype=np.float64)
        summary = np.zeros(12, dtype=np.float64)
        cfg_arr = np.array(
            [
                cfg.timestep_ps,
                cfg.temperature_k,
                gamma_t,
                zeta_r,
                cfg.launch_distance_a,
                cfg.stage_pull_velocity,
                cfg.stage_spring_k,
                cfg.stage_max_steps,
                cfg.release_steps,
                cfg.record_every,
                cfg.seed,
                h_trans,
                h_rot,
            ],
            dtype=np.float64,
        )
        moments = np.ascontiguousarray(eng.moments)
        com0 = np.ascontiguousarray(eng.com0)
        quat0 = np.ascontiguousarray(eng.quat0)

        n = int(
            eng.lib.gex_simulate(
                ctypes.c_void_p(eng.handle),
                _p(cfg_arr),
                _p(moments),
                _p(com0),
                _p(quat0),
                _p(esc),
                ctypes.c_double(total_mass),
                _p(frames_buf),
                n_expected,
                _p(summary),
            )
        )
        if n < 0:
            raise RuntimeError(
                f"native trajectory buffer too small ({n_expected} frames); "
                "raise record_every or lower release_steps"
            )

        recs = frames_buf[: n * stride].reshape(n, stride)
        traj = BindingTrajectory()
        for row in recs:
            coords = row[_FRAME_HEADER:].reshape(eng.n_lig, 3)
            traj.frames.append(
                TrajectoryFrame(
                    step=int(row[0]),
                    time_ps=float(row[1]),
                    com=[float(v) for v in row[2:5]],
                    quat=[float(v) for v in row[5:9]],
                    coords=[[float(c) for c in xyz] for xyz in coords],
                    u_total=float(row[9]),
                    e_coulomb=float(row[10]),
                    dg_gb_solvation=float(row[11]),
                    e_vdw=float(row[12]),
                    e_hbond=float(row[13]),
                    e_halogen=float(row[14]),
                    e_cation_pi=float(row[15]),
                    e_pi_stacking=float(row[16]),
                    dg_hydrophobic=float(row[17]),
                    temp_trans_k=float(row[18]),
                    temp_rot_k=float(row[19]),
                    com_displacement_a=float(np.linalg.norm(row[2:5] - eng.com0)),
                    min_contact_a=float(row[20]),
                )
            )

        kt = cfg.temperature_k * KB_KCAL_MOL_K
        rot_rates = zeta_r / eng.moments
        traj.diagnostics = {
            "backend": "native C++ (std::thread), float64 state",
            "n_frames": n,
            "total_mass_amu": round(total_mass, 3),
            "principal_moments_amu_a2": [round(float(v), 2) for v in eng.moments],
            "hydrodynamic_radius_a": round(radius, 4),
            "gamma_trans_per_ps": round(gamma_t, 4),
            "zeta_rot_amu_a2_per_ps": round(zeta_r, 1),
            "rot_rates_per_ps": [round(float(v), 2) for v in rot_rates],
            "rot_rate_times_dt": [round(float(v) * cfg.timestep_ps, 4) for v in rot_rates],
            "timestep_ps": cfg.timestep_ps,
            "release_steps": int(cfg.release_steps),
            "simulated_time_ps": round(int(cfg.release_steps) * cfg.timestep_ps, 4),
            "temperature_target_k": cfg.temperature_k,
            "stage_steps_used": int(summary[0]),
            "stage_max_pull_force": round(float(summary[2]), 2),
            "launch_displacement_a": round(float(summary[1]), 3),
            "mean_ke_trans_kcal_mol": round(float(summary[3]), 5),
            "mean_ke_rot_kcal_mol": round(float(summary[4]), 5),
            "expected_ke_per_3dof_kcal_mol": round(1.5 * kt, 5),
            "temp_trans_effective_k": round(float(summary[5]), 1),
            "temp_rot_effective_k": round(float(summary[6]), 1),
            "final_com_displacement_a": round(float(summary[7]), 3),
            "closest_contact_a": round(float(summary[8]), 3),
            "born_mirror_max_error_a": float(f"{eng.born_check:.3e}"),
            "fd_step_trans_a": h_trans,
            "fd_step_rot_rad": h_rot,
            "softcore_lambda": 1.0,
            "force_clipping": "none",
            "force_method": "central finite differences on the 6 rigid-body DOF",
            "rotational_integrator": "DLM symmetric free-top R-splitting (2nd order)",
            "switch_on_a": float(hamiltonian.switch_on),
            "nonbonded_cutoff_a": float(hamiltonian.nonbonded_cutoff),
            "contract": (
                "Boltzmann sampling of a fixed empirical scoring function at a nominal "
                "temperature; rigid ligand, rigid receptor. Not solvated binding kinetics."
            ),
        }
        return traj
    finally:
        if engine is None:
            eng.close()


# =======================================================================================
# WebAssembly support
# =======================================================================================


def pack_engine_blob(
    lig: BiophysicalAtomParams,
    rec: BiophysicalAtomParams,
    hamiltonian: BiophysicalHamiltonian,
) -> bytes:
    """Pack everything ``gex_create`` needs into one buffer, for the browser build.

    The browser gets the finished arrays rather than the code that computes them. All of
    that setup -- the HCT descreening integrals, the apo Born radii, the apo Still f_GB
    matrix, the apo SASA, the donor and sigma-hole axes, the ring frames -- already exists
    and is checked against the PyTorch reference on the Python side. Re-deriving it in
    JavaScript would be a second implementation to keep in step with the first, and a
    silent divergence there would show up as a believable animation of the wrong physics.

    Layout: ``4 x int32`` header (nl, np, n_lring, n_rring), the ``GexParams`` block, the
    float64 arrays in the order ``gex_create`` takes them **except the two np x np
    receptor matrices**, then the two uint8 flag arrays. Those matrices are omitted on
    purpose: they are 19 MB for a 1090-atom pocket and are a few milliseconds of O(N^2)
    arithmetic away from the receptor coordinates and apo Born radii, which are in the
    blob already, so ``gex_create_packed`` derives them on load.

    Returns:
        Bytes suitable for ``gex_create_packed``.
    """
    import struct

    eng = NativeEngine.build(lig, rec, hamiltonian)
    try:
        parts = [struct.pack("<4i", eng.n_lig, eng.n_rec, eng._n_lring, eng._n_rring)]
        parts.append(bytes(eng._params))
        # The last two entries are the np x np receptor matrices, which
        # gex_create_packed recomputes rather than downloading: 19 MB versus a few ms.
        for arr in eng._packed_arrays[:-2]:
            parts.append(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        for arr in eng._packed_flags:
            parts.append(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
        return b"".join(parts)
    finally:
        eng.close()


def build_wasm_module(out_dir: Path | str, *, optimize: str = "-O3") -> Path:
    """Compile the same engine to WebAssembly with Emscripten.

    The browser runs the identical C++ source as the native build, which is the whole
    argument for WebAssembly over a JavaScript port: there is no second implementation to
    drift. The build is single-threaded -- pthreads in WebAssembly need SharedArrayBuffer
    and COOP/COEP headers on the serving origin, and the engine already has a serial path.

    Args:
        out_dir: Directory to write ``gexdyn.js`` and ``gexdyn.wasm`` into.
        optimize: Optimisation flag passed to ``emcc``.

    Returns:
        Path to the generated loader ``gexdyn.js``.

    Raises:
        RuntimeError: If ``emcc`` is not on PATH, or the compile fails.
    """
    import shutil

    # em++ rather than emcc: with a .cpp input, emcc links without libc++ and fails on
    # `operator new` / `operator delete`, which is exactly what its own warning suggests.
    emcc = shutil.which("em++") or shutil.which("emcc")
    if emcc is None:
        raise RuntimeError(
            "em++ not found. Install Emscripten, e.g.\n"
            "  git clone https://github.com/emscripten-core/emsdk ~/emsdk\n"
            "  cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest\n"
            "  source ~/emsdk/emsdk_env.sh\n"
            "then re-run build_wasm_module()."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "gexdyn.js"
    exported = [
        "_gex_version",
        "_gex_create_packed",
        "_gex_destroy",
        "_gex_energy",
        "_gex_forces",
        "_gex_pose",
        "_gex_simulate",
        "_malloc",
        "_free",
    ]
    cmd = [
        emcc,
        str(_SRC),
        "-o",
        str(js),
        "-std=c++17",
        optimize,
        # Free on any browser that runs WebAssembly at all (SIMD has shipped since 2021).
        # Its benefit here was NOT cleanly measurable: successive runs on a loaded machine
        # gave 448, 389 and 531 ms for the same force evaluation, so the spread between
        # builds is smaller than the spread between runs. Do not quote a speedup for it.
        "-msimd128",
        "-ffp-contract=off",
        "-s",
        "MODULARIZE=1",
        "-s",
        "EXPORT_NAME=GexDyn",
        "-s",
        "ALLOW_MEMORY_GROWTH=1",
        "-s",
        "ENVIRONMENT=web,worker",
        "-s",
        f"EXPORTED_FUNCTIONS={json.dumps(exported)}",
        "-s",
        'EXPORTED_RUNTIME_METHODS=["HEAPU8","HEAPF64","ccall","cwrap"]',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not js.is_file():
        raise RuntimeError(f"emcc failed\ncommand: {' '.join(cmd)}\n{proc.stderr or proc.stdout}")
    return js
