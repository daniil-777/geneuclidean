#!/usr/bin/env python3
"""Auto-generated OpenFE RBFE driver for GenEuclidean perturbation network."""
from __future__ import annotations

import json
from pathlib import Path

def main() -> None:
    import openfe
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from rdkit import Chem

    here = Path(__file__).resolve().parent
    spec = json.loads((here / "rbfe_network.json").read_text(encoding="utf-8"))
    protein = openfe.ProteinComponent.from_pdb_file(spec["pocket_pdb"])
    solvent = openfe.SolventComponent()
    lig_map = {}
    for m in Chem.SDMolSupplier(str(here / spec["ligands_sdf"]), removeHs=False):
        if m is not None:
            name = m.GetProp("_Name")
            lig_map[name] = openfe.SmallMoleculeComponent.from_rdkit(m, name=name)

    settings = RelativeHybridTopologyProtocol.default_settings()
    protocol = RelativeHybridTopologyProtocol(settings=settings)
    print(f"Loaded OpenFE network: {len(lig_map)} ligands, {len(spec['edges'])} RBFE edges.")
    print("Protocol:", type(protocol).__name__, "| protein:", protein.name, "| solvent:", solvent)

if __name__ == "__main__":
    main()
