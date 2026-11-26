"""Pure-RDKit reaction-grounded makeable synthesis engine & retrosynthesis registry.

Scientific motivation:
    In 2025-2026 industrial CADD (Roche pRED, Enamine REAL Space, AstraZeneca AiZynthFinder,
    SynFormer), heuristic ``SA score`` alone is insufficient for medicinal chemists: generated
    analogs must map directly to robust, parallelizable 1-2 step coupling chemistry with
    explicit building blocks. This module encodes 18 curated high-yield MedChem reactions
    (Hartenfeller / Enamine REAL SMIRKS subset) with strict reagent/byproduct quarantine and
    provides a pluggable :data:`RETROSYNTHESIS_REGISTRY` for external planners such as
    ``AiZynthFinder``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from geneuclidean.cns.medchem_triage import triage_molecule
from geneuclidean.registry import Registry

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class MedChemReaction:
    """Single robust 2-component medicinal-chemistry coupling transformation."""

    name: str
    smirks: str
    category: str
    description: str


# Curated high-yield parallel MedChem reactions (Hartenfeller et al. / Enamine REAL core)
ROBUST_MEDCHEM_REACTIONS: tuple[MedChemReaction, ...] = (
    MedChemReaction(
        name="amide_coupling_acid_amine",
        smirks="[C:1](=[O:2])[OH].[N;H1,H2;!$(NC=O);!$(NS=O):3]>>[C:1](=[O:2])[N:3]",
        category="acylation",
        description="Carboxylic acid + primary/secondary amine -> amide (HATU/EDC)",
    ),
    MedChemReaction(
        name="amide_coupling_acyl_chloride",
        smirks="[C:1](=[O:2])[Cl].[N;H1,H2;!$(NC=O);!$(NS=O):3]>>[C:1](=[O:2])[N:3]",
        category="acylation",
        description="Acyl chloride + amine -> amide",
    ),
    MedChemReaction(
        name="sulfonamide_coupling",
        smirks="[S:1](=[O:2])(=[O:3])[Cl].[N;H1,H2;!$(NC=O);!$(NS=O):4]>>[S:1](=[O:2])(=[O:3])[N:4]",
        category="sulfonylation",
        description="Sulfonyl chloride + amine -> sulfonamide",
    ),
    MedChemReaction(
        name="reductive_amination_aldehyde",
        smirks="[#6:1][CH1:2]=[O].[N;H1,H2;!$(NC=O);!$(NS=O):3]>>[#6:1][CH2:2][N:3]",
        category="alkylation",
        description="Aldehyde + primary/secondary amine -> alkylated amine (NaBH(OAc)3)",
    ),
    MedChemReaction(
        name="suzuki_miyaura_biaryl",
        smirks="[c:1][Br,I].[c:2]B(O)O>>[c:1][c:2]",
        category="cross_coupling",
        description="Aryl bromide/iodide + aryl boronic acid -> biaryl (Pd)",
    ),
    MedChemReaction(
        name="buchwald_hartwig_amination",
        smirks="[c:1][Br,I].[N;H1,H2;!$(NC=O);!$(NS=O):2]>>[c:1][N:2]",
        category="cross_coupling",
        description="Aryl bromide/iodide + amine -> arylamine (Pd)",
    ),
    MedChemReaction(
        name="snar_heteroaryl_fluoride",
        smirks="[c;r6:1]([F,Cl]):[n:2].[N;H1,H2;!$(NC=O);!$(NS=O):3]>>[c:1]([N:3]):[n:2]",
        category="heteroaryl_substitution",
        description="2-Halopyridine/pyrimidine + amine -> amino-heterocycle (SNAr)",
    ),
    MedChemReaction(
        name="williamson_aryl_ether",
        smirks="[c:1][OH].[CX4:2][Br,Cl,I]>>[c:1][O][C:2]",
        category="etherification",
        description="Phenol + alkyl halide -> aryl alkyl ether",
    ),
    MedChemReaction(
        name="n_alkylation_halide",
        smirks="[N;H1,H2;!$(NC=O);!$(NS=O):1].[CX4:2][Br,I]>>[N:1][C:2]",
        category="alkylation",
        description="Amine + alkyl bromide/iodide -> N-alkylated amine",
    ),
    MedChemReaction(
        name="sonogashira_alkynylation",
        smirks="[c:1][Br,I].[CH1:2]#[C:3]>>[c:1][C:2]#[C:3]",
        category="cross_coupling",
        description="Aryl halide + terminal alkyne -> aryl alkyne (Pd/Cu)",
    ),
    MedChemReaction(
        name="cuaac_triazole_click",
        smirks="[#6:1][N:2]=[N+]=[N-].[CH1:3]#[C:4]>>[#6:1][n:2]1nn[c:4][c:3]1",
        category="click",
        description="Organic azide + terminal alkyne -> 1,4-disubstituted 1,2,3-triazole",
    ),
    MedChemReaction(
        name="urea_from_isocyanate",
        smirks="[#6:1][N:2]=[C:3]=[O:4].[N;H1,H2;!$(NC=O):5]>>[#6:1][NH:2][C:3](=[O:4])[N:5]",
        category="urea",
        description="Isocyanate + amine -> N,N'-disubstituted urea",
    ),
    MedChemReaction(
        name="esterification_acid_alcohol",
        smirks="[C:1](=[O:2])[OH].[OX2H1;!$(OC=O):3][#6:4]>>[C:1](=[O:2])[O:3][#6:4]",
        category="acylation",
        description="Carboxylic acid + alcohol -> ester",
    ),
    MedChemReaction(
        name="aryl_ether_mitsunobu",
        smirks="[c:1][OH].[CX4:2][OH]>>[c:1][O][C:2]",
        category="etherification",
        description="Phenol + aliphatic alcohol -> aryl ether (Mitsunobu)",
    ),
    MedChemReaction(
        name="heteroaryl_sulfonamide",
        smirks="[c:1][S:2](=[O:3])(=[O:4])[Cl].[NH1,NH2;R:5]>>[c:1][S:2](=[O:3])(=[O:4])[N:5]",
        category="sulfonylation",
        description="Aryl sulfonyl chloride + cyclic secondary amine -> sulfonamide",
    ),
    MedChemReaction(
        name="piperazide_acylation",
        smirks="[c:1][C:2](=[O:3])[OH].[NH1;R:4]1CCNCC1>>[c:1][C:2](=[O:3])[N:4]1CCNCC1",
        category="acylation",
        description="Aryl carboxylic acid + piperazine -> aryl piperazide",
    ),
)

_COMPILED_RXNS: dict[str, tuple[MedChemReaction, Any]] | None = None


def _get_compiled_reactions() -> dict[str, tuple[MedChemReaction, Any]]:
    global _COMPILED_RXNS
    if _COMPILED_RXNS is None:
        table: dict[str, tuple[MedChemReaction, Any]] = {}
        for rxn in ROBUST_MEDCHEM_REACTIONS:
            obj = AllChem.ReactionFromSmarts(rxn.smirks)  # type: ignore[attr-defined]
            if obj is not None:
                table[rxn.name] = (rxn, obj)
        _COMPILED_RXNS = table
    return _COMPILED_RXNS


@dataclass(frozen=True)
class MakeableCandidate:
    """Reaction-grounded analog carrying explicit 1-step synthetic provenance."""

    product_smiles: str
    reaction_name: str
    reaction_category: str
    reactant_a: str
    reactant_b: str
    reaction_smirks: str
    triage_passed: bool
    hygiene_gate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_smiles": self.product_smiles,
            "reaction_name": self.reaction_name,
            "reaction_category": self.reaction_category,
            "reactant_a": self.reactant_a,
            "reactant_b": self.reactant_b,
            "reaction_smirks": self.reaction_smirks,
            "triage_passed": self.triage_passed,
            "hygiene_gate": round(self.hygiene_gate, 4),
        }


def enumerate_makeable_analogs(
    core_smiles: str,
    building_blocks: Sequence[str],
    *,
    reactions: Sequence[str] | None = None,
    max_products: int = 500,
    triage_filter: bool = True,
) -> list[MakeableCandidate]:
    """Enumerate 1-step makeable analogs coupling ``core_smiles`` with ``building_blocks``."""
    core_mol = Chem.MolFromSmiles(core_smiles.strip())
    if core_mol is None:
        raise ValueError(f"Invalid core_smiles: {core_smiles!r}")
    core_can = Chem.MolToSmiles(core_mol, canonical=True, isomericSmiles=True)

    table = _get_compiled_reactions()
    selected = (
        [table[r] for r in reactions if r in table]
        if reactions is not None
        else list(table.values())
    )

    bb_mols: list[tuple[str, Chem.Mol]] = []
    for raw_bb in building_blocks:
        m = Chem.MolFromSmiles(raw_bb.strip())
        if m is not None:
            bb_mols.append((Chem.MolToSmiles(m, canonical=True, isomericSmiles=True), m))

    seen_products: set[str] = set()
    candidates: list[MakeableCandidate] = []

    for rxn_spec, rxn_obj in selected:
        for bb_smi, bb_mol in bb_mols:
            for mol_a, smi_a, mol_b, smi_b in (
                (core_mol, core_can, bb_mol, bb_smi),
                (bb_mol, bb_smi, core_mol, core_can),
            ):
                try:
                    prods = rxn_obj.RunReactants((mol_a, mol_b))
                except Exception:
                    continue
                for prod_tuple in prods:
                    if not prod_tuple:
                        continue
                    pmol = prod_tuple[0]
                    try:
                        Chem.SanitizeMol(pmol)
                        psmi = Chem.MolToSmiles(pmol, canonical=True, isomericSmiles=True)
                    except Exception:
                        continue
                    if not psmi or "." in psmi or psmi in seen_products:
                        continue
                    tr = triage_molecule(psmi)
                    if triage_filter and not tr.triage_pass:
                        continue
                    seen_products.add(psmi)
                    candidates.append(
                        MakeableCandidate(
                            product_smiles=psmi,
                            reaction_name=rxn_spec.name,
                            reaction_category=rxn_spec.category,
                            reactant_a=smi_a,
                            reactant_b=smi_b,
                            reaction_smirks=rxn_spec.smirks,
                            triage_passed=bool(tr.triage_pass),
                            hygiene_gate=float(tr.hygiene_gate),
                        )
                    )
                    if len(candidates) >= int(max_products):
                        return candidates
    return candidates


# Built-in 1-step retrosynthesis bond-disconnection patterns (fast offline screen)
_RETRO_BOND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("amide_bond", "[CX3;!R](=O)-!@[NX3]"),
    ("sulfonamide_bond", "[SX4](=O)(=O)-!@[NX3]"),
    ("biaryl_suzuki", "[c;R]-!@[c;R]"),
    ("aryl_ether", "[c;R]-[OX2]-!@[CX4]"),
    ("arylamine_buchwald", "[c;R]-!@[NX3;!$(NC=O)]"),
    ("ester_bond", "[CX3;!R](=O)-[OX2;!R]-[#6]"),
    ("triazole_click", "n1nncc1"),
    ("urea_bond", "[NX3]-[CX3](=O)-[NX3]"),
)


def match_one_step_retrosynthesis(target_smiles: str) -> dict[str, Any]:
    """Identify standard 1-step MedChem retrosynthetic disconnections in ``target_smiles``."""
    mol = Chem.MolFromSmiles(target_smiles.strip())
    if mol is None:
        return {"valid": False, "is_solved": False, "matched_disconnections": []}
    matched: list[str] = []
    for name, smarts in _RETRO_BOND_PATTERNS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            matched.append(name)
    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "is_solved": len(matched) > 0,
        "n_disconnections": len(matched),
        "matched_disconnections": matched,
        "backend": "template_1step",
    }


RetrosynthesisFn = Callable[[str], dict[str, Any]]
RETROSYNTHESIS_REGISTRY: Registry[RetrosynthesisFn] = Registry("retrosynthesis_backends")
RETROSYNTHESIS_REGISTRY.register("template_1step", match_one_step_retrosynthesis)


def register_retrosynthesis_backend(
    name: str, fn: RetrosynthesisFn, *, override: bool = False
) -> RetrosynthesisFn:
    """Register an external retrosynthesis planner (e.g., AstraZeneca ``AiZynthFinder``)."""
    return RETROSYNTHESIS_REGISTRY.register(name, fn, override=override)


__all__ = [
    "RETROSYNTHESIS_REGISTRY",
    "ROBUST_MEDCHEM_REACTIONS",
    "MakeableCandidate",
    "MedChemReaction",
    "enumerate_makeable_analogs",
    "match_one_step_retrosynthesis",
    "register_retrosynthesis_backend",
]
