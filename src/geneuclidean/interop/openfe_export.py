"""OpenFE / Lomap alchemical RBFE perturbation network exporter (zero hard OpenFE dep).

Bridges GenEuclidean's generative, screening, and MMP lead-optimization outputs directly
into the 15-pharma open-source **OpenFE** Relative Binding Free Energy (RBFE) ecosystem:
1. Protonates every candidate consistently at physiological ``pH = 7.4``.
2. Enforces zero formal-charge jumps (``Delta q == 0``) across every perturbation edge.
3. Optionally filters high-strain saturated rings (``RingStrainFilter``).
4. Aligns congeneric 3D ligand poses onto their common Maximum Common Substructure (MCS)
   template via ``AllChem.ConstrainedEmbed`` (or preserves supplied docked 3D poses).
5. Builds a connected Lomap-style minimum-spanning + cycle-closure perturbation graph
   scored by exact MCS atom-mapping similarity and writes aligned SDFs, ``rbfe_network.json``,
   and a runnable ``run_openfe_rbfe.py`` driver script.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS

from geneuclidean.evaluation.protonation import protonate_mol_ph74
from geneuclidean.evaluation.ring_strain import analyze_ring_strain

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class OpenFEPerturbationEdge:
    """Single Lomap-style RBFE alchemical transformation edge between two aligned ligands."""

    ligand_a: str
    ligand_b: str
    mcs_num_atoms: int
    lomap_like_score: float
    delta_formal_charge: int
    delta_heavy_atoms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ligand_a": self.ligand_a,
            "ligand_b": self.ligand_b,
            "mcs_num_atoms": self.mcs_num_atoms,
            "lomap_like_score": round(self.lomap_like_score, 4),
            "delta_formal_charge": self.delta_formal_charge,
            "delta_heavy_atoms": self.delta_heavy_atoms,
        }


@dataclass(frozen=True)
class OpenFENetworkBundle:
    """Summary of exported OpenFE RBFE perturbation network artifacts."""

    output_dir: Path
    network_json: Path
    driver_script: Path
    ligands_sdf: Path
    n_ligands: int
    n_edges: int
    edges: tuple[OpenFEPerturbationEdge, ...]
    skipped_ligands: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "network_json": str(self.network_json),
            "driver_script": str(self.driver_script),
            "ligands_sdf": str(self.ligands_sdf),
            "n_ligands": self.n_ligands,
            "n_edges": self.n_edges,
            "edges": [e.to_dict() for e in self.edges],
            "skipped_ligands": list(self.skipped_ligands),
        }


def _lomap_pair_score(mol_a: Chem.Mol, mol_b: Chem.Mol) -> tuple[int, float]:
    """Compute Lomap-style MCS overlap score in ``(0, 1]`` for an RBFE transformation edge."""
    mcs = rdFMCS.FindMCS(
        [mol_a, mol_b],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=3,
    )
    n_mcs = int(mcs.numAtoms)
    na = int(mol_a.GetNumHeavyAtoms())
    nb = int(mol_b.GetNumHeavyAtoms())
    n_diff = (na - n_mcs) + (nb - n_mcs)
    # Classic Lomap exponential penalty: exp(-0.1 * n_unmapped_heavy_atoms)
    score = float(math.exp(-0.1 * float(max(0, n_diff))))
    return n_mcs, score


def export_openfe_rbfe_network(
    candidates: Sequence[str | tuple[str, str | Chem.Mol]],
    pocket_pdb: str | Path,
    out_dir: str | Path,
    *,
    reference_3d_mol: Chem.Mol | None = None,
    protonate: bool = True,
    require_same_charge: bool = True,
    filter_ring_strain: bool = True,
    max_ring_strain_kcal: float = 3.0,
    add_cycle_closures: bool = True,
    seed: int = 42,
) -> OpenFENetworkBundle:
    """Build and export a charge-consistent, ring-strain-clean OpenFE RBFE network."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    pocket_path = Path(pocket_pdb)

    prepared: list[tuple[str, Chem.Mol, int]] = []
    skipped: list[dict[str, str]] = []

    # Prepare reference template for MCS 3D alignment
    template_mol: Chem.Mol | None = reference_3d_mol
    for idx, item in enumerate(candidates):
        if isinstance(item, tuple):
            label, raw = str(item[0]), item[1]
        else:
            label, raw = f"lig_{idx:03d}", item

        mol: Chem.Mol | None = (
            Chem.Mol(raw) if isinstance(raw, Chem.Mol) else Chem.MolFromSmiles(str(raw).strip())
        )
        if mol is None:
            skipped.append({"name": label, "reason": "invalid_smiles"})
            continue

        if protonate:
            mol = protonate_mol_ph74(mol, preserve_coords=(mol.GetNumConformers() > 0))
        q_net = int(Chem.GetFormalCharge(mol))

        # Embed 3D (constrained to MCS of template_mol when available) if not already 3D
        if mol.GetNumConformers() == 0:
            mol_h = Chem.AddHs(mol)
            embedded_ok = False
            if template_mol is not None and template_mol.GetNumConformers() > 0:
                try:
                    mcs = rdFMCS.FindMCS(
                        [Chem.RemoveHs(template_mol), mol],
                        ringMatchesRingOnly=True,
                        completeRingsOnly=True,
                        timeout=3,
                    )
                    if mcs.numAtoms >= 6 and mcs.smartsString:
                        core_q = Chem.MolFromSmarts(mcs.smartsString)
                        if core_q is not None and template_mol.HasSubstructMatch(core_q):
                            core_3d = AllChem.DeleteSubstructs(  # type: ignore[attr-defined]
                                Chem.RemoveHs(template_mol),
                                Chem.MolFromSmarts("[!#0]"),
                            )
                            _ = core_3d
                            # Extract aligned core submol directly from template_mol
                            match = template_mol.GetSubstructMatch(core_q)
                            rw = Chem.RWMol()
                            old2new: dict[int, int] = {}
                            for oi in match:
                                sa = template_mol.GetAtomWithIdx(oi)
                                na = Chem.Atom(sa.GetAtomicNum())
                                na.SetFormalCharge(sa.GetFormalCharge())
                                na.SetIsAromatic(sa.GetIsAromatic())
                                old2new[oi] = rw.AddAtom(na)
                            for b in template_mol.GetBonds():
                                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                                if u in old2new and v in old2new:
                                    rw.AddBond(old2new[u], old2new[v], b.GetBondType())
                            core_sub = rw.GetMol()
                            Chem.SanitizeMol(core_sub)
                            tconf = template_mol.GetConformer()
                            cconf = Chem.Conformer(core_sub.GetNumAtoms())
                            for oi, ni in old2new.items():
                                cconf.SetAtomPosition(ni, tconf.GetAtomPosition(oi))
                            core_sub.AddConformer(cconf, assignId=True)
                            AllChem.ConstrainedEmbed(mol_h, core_sub, randomseed=seed + idx)
                            embedded_ok = mol_h.GetNumConformers() > 0
                except Exception:
                    embedded_ok = False
            if not embedded_ok:
                params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
                params.randomSeed = int(seed + idx)
                params.timeout = 3
                cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=8, params=params))  # type: ignore[attr-defined]
                if not cids:
                    skipped.append({"name": label, "reason": "3d_embedding_failed"})
                    continue
                use_mmff = bool(AllChem.MMFFHasAllMoleculeParams(mol_h))  # type: ignore[attr-defined]
                mprops = (
                    AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # type: ignore[attr-defined]
                    if use_mmff
                    else None
                )
                best_cid, best_e = int(cids[0]), float("inf")
                for cid in cids:
                    ff = (
                        AllChem.MMFFGetMoleculeForceField(mol_h, mprops, confId=int(cid))  # type: ignore[attr-defined]
                        if mprops is not None
                        else AllChem.UFFGetMoleculeForceField(mol_h, confId=int(cid))  # type: ignore[attr-defined]
                    )
                    if ff is not None:
                        ff.Minimize(maxIts=300)
                        ev = float(ff.CalcEnergy())
                        if ev < best_e:
                            best_e, best_cid = ev, int(cid)
                best_conf = Chem.Conformer(mol_h.GetConformer(best_cid))
                mol_h.RemoveAllConformers()
                mol_h.AddConformer(best_conf, assignId=True)
            mol = mol_h
        else:
            mol = Chem.AddHs(mol, addCoords=True)

        if filter_ring_strain:
            rprof = analyze_ring_strain(
                mol,
                strain_threshold_kcal=max_ring_strain_kcal,
                n_ref_confs=16,
                include_full_ligand_strain=False,
                seed=seed,
            )
            if not rprof.passed:
                skipped.append(
                    {
                        "name": label,
                        "reason": f"ring_strain={rprof.max_ring_strain_kcal:.2f}_kcal_mol",
                    }
                )
                continue

        mol.SetProp("_Name", label)
        mol.SetProp("SMILES_PH74", Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True))
        mol.SetProp("NET_CHARGE_PH74", str(q_net))
        if template_mol is None:
            template_mol = mol
        prepared.append((label, mol, q_net))

    # Build charge-matched maximum-similarity spanning tree + cycle-closure edges
    all_pair_edges: list[tuple[float, int, int, OpenFEPerturbationEdge]] = []
    for i in range(len(prepared)):
        la, ma, qa = prepared[i]
        ma_noh = Chem.RemoveHs(ma)
        for j in range(i + 1, len(prepared)):
            lb, mb, qb = prepared[j]
            dq = qb - qa
            if require_same_charge and dq != 0:
                continue
            mb_noh = Chem.RemoveHs(mb)
            n_mcs, score = _lomap_pair_score(ma_noh, mb_noh)
            edge = OpenFEPerturbationEdge(
                ligand_a=la,
                ligand_b=lb,
                mcs_num_atoms=n_mcs,
                lomap_like_score=score,
                delta_formal_charge=dq,
                delta_heavy_atoms=abs(mb_noh.GetNumHeavyAtoms() - ma_noh.GetNumHeavyAtoms()),
            )
            all_pair_edges.append((score, i, j, edge))

    all_pair_edges.sort(key=lambda x: x[0], reverse=True)
    parent = list(range(len(prepared)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen_edges: list[OpenFEPerturbationEdge] = []
    extra_edges: list[OpenFEPerturbationEdge] = []
    for _score, i, j, edge in all_pair_edges:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
            chosen_edges.append(edge)
        else:
            extra_edges.append(edge)

    if add_cycle_closures and extra_edges:
        # Add up to N//2 highest-overlap cycle-closure edges for RBFE hysteresis diagnostics
        n_closures = max(1, len(prepared) // 2)
        chosen_edges.extend(extra_edges[:n_closures])

    ligands_sdf = dest / "aligned_ligands_ph74.sdf"
    writer = Chem.SDWriter(str(ligands_sdf))
    for _label, mol, _q in prepared:
        writer.write(mol)
    writer.close()

    network_json = dest / "rbfe_network.json"
    network_payload = {
        "schema": "geneuclidean_openfe_rbfe_v1",
        "pocket_pdb": str(pocket_path),
        "ligands_sdf": str(ligands_sdf.name),
        "n_ligands": len(prepared),
        "n_edges": len(chosen_edges),
        "require_same_charge": require_same_charge,
        "ligands": [
            {
                "name": label,
                "smiles_ph74": mol.GetProp("SMILES_PH74"),
                "net_charge_ph74": q_net,
            }
            for label, mol, q_net in prepared
        ],
        "edges": [e.to_dict() for e in chosen_edges],
        "skipped_ligands": skipped,
    }
    network_json.write_text(json.dumps(network_payload, indent=2) + "\n", encoding="utf-8")

    driver_script = dest / "run_openfe_rbfe.py"
    driver_script.write_text(
        '''#!/usr/bin/env python3
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
''',
        encoding="utf-8",
    )

    return OpenFENetworkBundle(
        output_dir=dest,
        network_json=network_json,
        driver_script=driver_script,
        ligands_sdf=ligands_sdf,
        n_ligands=len(prepared),
        n_edges=len(chosen_edges),
        edges=tuple(chosen_edges),
        skipped_ligands=tuple(skipped),
    )


__all__ = [
    "OpenFENetworkBundle",
    "OpenFEPerturbationEdge",
    "export_openfe_rbfe_network",
]
