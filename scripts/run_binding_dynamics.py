#!/usr/bin/env python
"""Run a rigid-body Langevin association trajectory and write it as JSON.

This is the entry point that produced the trajectories quoted in the biophysics handoff.
It exists so those numbers have a command behind them: a figure in a document that no
script reproduces is an anecdote, not a result.

Example, the 120 ps association run from outside the protein::

    python scripts/run_binding_dynamics.py \\
        --pocket data/raw/casf2016_extra/3arv/3arv_pocket.pdb \\
        --ligand data/raw/casf2016_extra/3arv/3arv_ligand.sdf \\
        --release-steps 60000 --launch 12.0 --stage-max-steps 14000 \\
        --record-every 50 --seed 11 --out traj_3arv_far.json

The pocket and ligand are required arguments rather than a built-in lookup table, because
``data/raw`` is gitignored and will not exist in a fresh clone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pocket", required=True, type=Path, help="pocket PDB")
    parser.add_argument(
        "--ligand", required=True, type=Path, help="ligand SDF, with explicit hydrogens"
    )
    parser.add_argument("--out", required=True, type=Path, help="trajectory JSON to write")
    parser.add_argument("--release-steps", type=int, default=20000, help="recorded Langevin steps")
    parser.add_argument(
        "--launch", type=float, default=8.0, help="staged launch displacement, Angstrom"
    )
    parser.add_argument("--stage-max-steps", type=int, default=4000, help="cap on steered-MD steps")
    parser.add_argument(
        "--stage-pull-velocity", type=float, default=4.0, help="pull speed, Angstrom/ps"
    )
    parser.add_argument(
        "--record-every", type=int, default=25, help="steps between recorded frames"
    )
    parser.add_argument("--timestep", type=float, default=0.002, help="timestep, ps")
    parser.add_argument("--temperature", type=float, default=300.0, help="temperature, K")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed; runs are reproducible")
    parser.add_argument(
        "--backend",
        choices=("auto", "native", "python"),
        default="auto",
        help="auto prefers the C++ engine and falls back to PyTorch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one trajectory and write it to disk.

    Returns:
        Process exit status: 0 on success, 1 if an input file is missing or unparseable.
    """
    args = build_parser().parse_args(argv)

    for path in (args.pocket, args.ligand):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 1

    from rdkit import Chem

    from geneuclidean.biophysics.binding_dynamics import (
        BindingDynamicsConfig,
        make_dynamics_hamiltonian,
        simulate_binding,
    )
    from geneuclidean.biophysics.forcefield import (
        parameterize_ligand_biophysics,
        parameterize_pocket_biophysics,
    )

    # removeHs=False: the H-bond, donor-axis and Gasteiger-charge machinery needs the
    # explicit hydrogens the SDF carries. Stripping them silently degrades the physics.
    mol = Chem.MolFromMolFile(str(args.ligand), removeHs=False, sanitize=True)
    if mol is None:
        print(f"RDKit could not parse {args.ligand}", file=sys.stderr)
        return 1

    lig = parameterize_ligand_biophysics(mol)
    rec = parameterize_pocket_biophysics(str(args.pocket))
    # Not the default Hamiltonian: the shipped 10-12 A switching window leaves a spurious
    # attractive shell right where an approach would be animated.
    hamiltonian = make_dynamics_hamiltonian()

    print(
        f"{args.ligand.stem}: ligand {lig.coords.shape[0]} atoms, "
        f"pocket {rec.coords.shape[0]} atoms",
        flush=True,
    )

    config = BindingDynamicsConfig(
        temperature_k=args.temperature,
        timestep_ps=args.timestep,
        launch_distance_a=args.launch,
        stage_max_steps=args.stage_max_steps,
        stage_pull_velocity=args.stage_pull_velocity,
        release_steps=args.release_steps,
        record_every=args.record_every,
        seed=args.seed,
    )

    started = time.perf_counter()
    traj = simulate_binding(lig, rec, hamiltonian, config=config, backend=args.backend)
    wall = time.perf_counter() - started

    for key, value in traj.diagnostics.items():
        print(f"  {key:34s} {value}")
    print(f"  {'wall_clock_s':34s} {wall:.1f}")

    # Staging is the step most likely to fail silently on a buried pocket: it can stall
    # against the wall and leave a strained pose that merely relaxes on release, which
    # looks like a trajectory and is not an association event.
    launched = float(traj.diagnostics.get("launch_displacement_a", 0.0))
    if launched < 0.8 * args.launch:
        print(
            f"\n  WARNING: staging reached only {launched:.2f} A of the {args.launch:.2f} A "
            "requested. The ligand is probably jammed against the pocket wall; check the "
            "first frames' u_total before describing this run as an association.",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(traj.to_dict()))
    print(
        f"\n  wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(traj.frames)} frames)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
