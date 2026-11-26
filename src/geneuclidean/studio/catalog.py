"""Authoritative GenEuclidean Circuit Studio block, port, signal & preset catalog.

Zero non-stdlib imports at module load time so ``import geneuclidean`` and
``import geneuclidean.studio`` preserve sub-15 ms cold-import latency with zero
eager ``torch``, ``transformers``, ``sklearn``, or ``rdkit`` imports.

All 10 typed signals, 8 categories, 36 real GenEuclidean blocks, illegal-wiring
remediation hints, and 5 flagship presets conform byte-for-byte to the shared
``CircuitGraphV1`` contract in ``docs/circuit_studio_architecture.md``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

SCHEMA_VERSION: Literal["geneuclidean.circuit.v1"] = "geneuclidean.circuit.v1"

SignalType = Literal[
    "POCKET_3D",
    "SE3_GRAPH",
    "LATENT_VEC",
    "MOL_BATCH",
    "POSE_3D",
    "ORACLE_SCORES",
    "UNCERTAINTY_DIST",
    "REAGENTS",
    "RBFE_GRAPH",
    "PROVENANCE_REPORT",
]

BlockCategoryId = Literal[
    "inputs",
    "encoders",
    "generators",
    "screening",
    "active_learning",
    "physics_audit",
    "leadopt_admet",
    "outputs",
]

EdgeKind = Literal["forward", "feedback"]
ExecutionTier = Literal["zero_torch", "optional_torch", "v51_al"]
PortDirection = Literal["input", "output"]
ParamType = Literal["number", "select", "boolean", "string"]


# ---------------------------------------------------------------------------
# 1. Typed Signal Grammar (10 Semantic Signals) & Category Metadata (8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """Visual & runtime contract metadata for one typed circuit signal."""

    id: SignalType
    label: str
    dark_hex: str
    light_hex: str
    wire_style: str
    python_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "darkHex": self.dark_hex,
            "lightHex": self.light_hex,
            "wireStyle": self.wire_style,
            "pythonType": self.python_type,
        }


SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec(
        id="POCKET_3D",
        label="3D Pocket",
        dark_hex="#38BDF8",
        light_hex="#0284C7",
        wire_style="solid",
        python_type="ge.Pocket | Sequence[ge.Pocket]",
    ),
    SignalSpec(
        id="SE3_GRAPH",
        label="SE(3) Graph",
        dark_hex="#818CF8",
        light_hex="#4F46E5",
        wire_style="solid",
        python_type="PocketEncoderOutput",
    ),
    SignalSpec(
        id="LATENT_VEC",
        label="Latent Vector",
        dark_hex="#A78BFA",
        light_hex="#7C3AED",
        wire_style="solid",
        python_type="NDArray[float16/32] | Tensor",
    ),
    SignalSpec(
        id="MOL_BATCH",
        label="Molecules",
        dark_hex="#34D399",
        light_hex="#059669",
        wire_style="solid",
        python_type="list[str] | DesignResult | list[ScreeningHit]",
    ),
    SignalSpec(
        id="POSE_3D",
        label="3D Pose",
        dark_hex="#2DD4BF",
        light_hex="#0D9488",
        wire_style="solid",
        python_type="list[DockingPoseResult] | list[Chem.Mol]",
    ),
    SignalSpec(
        id="ORACLE_SCORES",
        label="Oracle Scores",
        dark_hex="#FBBF24",
        light_hex="#D97706",
        wire_style="solid_or_feedback_dashed",
        python_type="dict[str, dict[str, float]]",
    ),
    SignalSpec(
        id="UNCERTAINTY_DIST",
        label="Uncertainty Dist",
        dark_hex="#F472B6",
        light_hex="#DB2777",
        wire_style="solid",
        python_type="SurrogatePrediction",
    ),
    SignalSpec(
        id="REAGENTS",
        label="Reagents",
        dark_hex="#FB923C",
        light_hex="#EA580C",
        wire_style="solid",
        python_type="dict[core_smiles, building_blocks, reactions]",
    ),
    SignalSpec(
        id="RBFE_GRAPH",
        label="RBFE Network",
        dark_hex="#FB7185",
        light_hex="#E11D48",
        wire_style="solid",
        python_type="OpenFENetworkBundle",
    ),
    SignalSpec(
        id="PROVENANCE_REPORT",
        label="Provenance",
        dark_hex="#CBD5E1",
        light_hex="#475569",
        wire_style="solid",
        python_type="PoseAuditReport | CredibilityDossier | dict[str, Any]",
    ),
)

VALID_SIGNAL_TYPES: frozenset[str] = frozenset(spec.id for spec in SIGNAL_SPECS)


@dataclass(frozen=True)
class CategorySpec:
    """Block dock category descriptor."""

    id: BlockCategoryId
    label: str
    accent_hex: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "accentHex": self.accent_hex,
            "order": self.order,
        }


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(id="inputs", label="Inputs & Structures", accent_hex="#38BDF8", order=1),
    CategorySpec(id="encoders", label="SE(3) / 3D Neural Encoders", accent_hex="#818CF8", order=2),
    CategorySpec(
        id="generators", label="De Novo Generators & Steering", accent_hex="#34D399", order=3
    ),
    CategorySpec(
        id="screening", label="Virtual Screening & Retrieval", accent_hex="#A78BFA", order=4
    ),
    CategorySpec(id="active_learning", label="Active Learning DMTA", accent_hex="#F472B6", order=5),
    CategorySpec(
        id="physics_audit",
        label="3D Physics & 7-Axis Pose Reality",
        accent_hex="#2DD4BF",
        order=6,
    ),
    CategorySpec(
        id="leadopt_admet", label="Lead-Opt, ADMET & Synthesis", accent_hex="#FBBF24", order=7
    ),
    CategorySpec(id="outputs", label="Outputs & Open-Pharma Export", accent_hex="#FB7185", order=8),
)

REMEDIATION_HINTS: dict[tuple[str, str], str] = {
    ("POCKET_3D", "SE3_GRAPH"): (
        "Insert PaiNN-SE(3), EGNN-3D, or E3NN Irreps to encode the 3D pocket "
        "into an equivariant graph."
    ),
    ("MOL_BATCH", "POSE_3D"): (
        "Insert AutoDock Vina 3D (or Boltz-2 / AF3 Co-Fold) to generate 3D ligand "
        "poses before geometry auditing."
    ),
    ("MOL_BATCH", "LATENT_VEC"): (
        "Insert DrugCLIP Encoder or IBM MMELON (84M) to embed SMILES into L2-normalized vectors."
    ),
    ("REAGENTS", "MOL_BATCH"): (
        "Insert 16-Reaction Makeable & Retro or Combinatorial Thompson Sampler "
        "to couple reagents into products."
    ),
    ("ORACLE_SCORES", "UNCERTAINTY_DIST"): (
        "Insert Tanimoto-Kernel GP or Ensemble Surrogate to fit a predictive "
        "posterior with epistemic uncertainty."
    ),
    ("POSE_3D", "RBFE_GRAPH"): (
        "Wire POSE_3D or MOL_BATCH into OpenFE RBFE Exporter to build a "
        "charge-matched (Δq = 0) Lomap graph."
    ),
}


def get_mismatch_remediation(source_signal: str, target_signal: str) -> str:
    """Return an actionable Apple-style remediation hint for an illegal signal pair."""
    hint = REMEDIATION_HINTS.get((source_signal, target_signal))
    if hint is not None:
        return hint
    return (
        f"Signal mismatch: {source_signal} cannot connect to {target_signal}. "
        f"Look for pulsing {source_signal}-colored sockets."
    )


# ---------------------------------------------------------------------------
# 2. Port, Parameter & Block Schema Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortSpec:
    """Typed input or output socket definition on a GenEuclidean circuit block."""

    id: str
    label: str
    signal_type: SignalType
    direction: PortDirection
    required: bool = False
    multi: bool = False
    accepts_feedback: bool = False
    shape_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "signalType": self.signal_type,
            "direction": self.direction,
            "required": self.required,
            "multi": self.multi if self.direction == "input" else True,
            "acceptsFeedback": self.accepts_feedback,
        }
        if self.shape_hint is not None:
            out["shapeHint"] = self.shape_hint
        return out


@dataclass(frozen=True)
class ParamSpec:
    """Tunable hyperparameter schema on a circuit block."""

    key: str
    label: str
    type: ParamType
    default: int | float | str | bool
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    options: tuple[str | int, ...] | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
        }
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.step is not None:
            out["step"] = self.step
        if self.unit is not None:
            out["unit"] = self.unit
        if self.options is not None:
            out["options"] = list(self.options)
        if self.description is not None:
            out["description"] = self.description
        return out


@dataclass(frozen=True)
class BlockDefinition:
    """Authoritative specification for one real GenEuclidean circuit block."""

    block_type: str
    name: str
    short_name: str
    category: BlockCategoryId
    python_symbol: str
    execution_tier: ExecutionTier
    badge: str
    summary: str
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    params: tuple[ParamSpec, ...]

    def input_map(self) -> dict[str, PortSpec]:
        return {p.id: p for p in self.inputs}

    def output_map(self) -> dict[str, PortSpec]:
        return {p.id: p for p in self.outputs}

    def param_map(self) -> dict[str, ParamSpec]:
        return {p.key: p for p in self.params}

    def default_params(self) -> dict[str, int | float | str | bool]:
        return {p.key: p.default for p in self.params}

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockType": self.block_type,
            "name": self.name,
            "shortName": self.short_name,
            "category": self.category,
            "pythonSymbol": self.python_symbol,
            "executionTier": self.execution_tier,
            "badge": self.badge,
            "summary": self.summary,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
        }


# ---------------------------------------------------------------------------
# 3. Complete 36-Block Authoritative GenEuclidean Library Catalog
# ---------------------------------------------------------------------------

_ALL_28_TARGETS: tuple[str, ...] = (
    "DRD1",
    "DRD2",
    "DRD3",
    "VMAT2",
    "ADRA2A",
    "HRH3",
    "PDE10A",
    "CHRM4",
    "HTR2A",
    "SERT",
    "TAAR1",
    "DAT",
    "ACHE",
    "MAOB",
    "NLRP3",
    "OGA",
    "GSK3B",
    "BACE1",
    "CHRM1",
    "HTR1A",
    "GLYT1",
    "GPR52",
    "MGLL",
    "PDE1B",
    "GABRA5",
    "GAT1",
    "ADORA2A",
    "GRM5",
)

_PROPERTY_PROFILE_OPTIONS: tuple[str, ...] = (
    "cns_mpo_v2",
    "oral_ro5",
    "kinase_selectivity",
    "beyond_ro5",
)

_REACTION_OPTIONS: tuple[str, ...] = (
    "amide_coupling_acid_amine",
    "sulfonamide_coupling",
    "reductive_amination_aldehyde",
    "suzuki_miyaura_biaryl",
    "buchwald_hartwig_amination",
    "snar_heteroaryl_fluoride",
    "cuaac_triazole_click",
    "urea_from_isocyanate",
)

BLOCK_DEFINITIONS: tuple[BlockDefinition, ...] = (
    # =====================================================================
    # Category 1 — Inputs & Structures (5 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="protein_pocket",
        name="Protein Pocket",
        short_name="Protein Pocket",
        category="inputs",
        python_symbol="geneuclidean.sdk.pocket.Pocket.from_target",
        execution_tier="zero_torch",
        badge="28 Targets",
        summary=(
            "Loads any of the 28 curated human CNS/Neuro 3D pockets (preserving catalytic "
            "cofactors FAD/ADP/Zn/Mg) or a local PDB/mmCIF structure."
        ),
        inputs=(),
        outputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="output",
                multi=True,
                shape_hint="PocketFeatures[N×3, N×52]",
            ),
        ),
        params=(
            ParamSpec(
                key="target_id",
                label="Target ID",
                type="select",
                default="DRD1",
                options=_ALL_28_TARGETS,
            ),
            ParamSpec(
                key="pocket_cutoff",
                label="Pocket Cutoff",
                type="number",
                default=10.0,
                min=6.0,
                max=16.0,
                step=0.5,
                unit="Å",
            ),
            ParamSpec(
                key="max_atoms",
                label="Max Heavy Atoms",
                type="number",
                default=384,
                min=128,
                max=512,
                step=32,
            ),
            ParamSpec(
                key="keep_cofactors",
                label="Preserve Cofactors (FAD/ADP/Zn/Mg)",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="multiconformer_ensemble",
        name="Multi-Conformer Ensemble",
        short_name="Multi-Conformer",
        category="inputs",
        python_symbol="geneuclidean.sdk.pocket.Pocket.from_target",
        execution_tier="zero_torch",
        badge="K-Conf + Off",
        summary=(
            "Bundles K experimental holo/apo conformations of an on-target pocket plus an "
            "optional off-target paralog counter-pocket for contrastive selectivity steering."
        ),
        inputs=(),
        outputs=(
            PortSpec(
                id="on_pockets",
                label="On-Target Ensemble",
                signal_type="POCKET_3D",
                direction="output",
                multi=True,
                shape_hint="Ensemble[K×Pocket]",
            ),
            PortSpec(
                id="off_pocket",
                label="Off-Target Counter",
                signal_type="POCKET_3D",
                direction="output",
                multi=True,
                shape_hint="CounterPocket",
            ),
        ),
        params=(
            ParamSpec(
                key="on_target_id",
                label="On-Target ID",
                type="select",
                default="DRD1",
                options=("DRD1", "ACHE", "MAOB", "BACE1", "CHRM4", "SERT", "GRM5", "GABRA5"),
            ),
            ParamSpec(
                key="n_conformers",
                label="Conformers (K)",
                type="number",
                default=2,
                min=1,
                max=4,
                step=1,
            ),
            ParamSpec(
                key="off_target_id",
                label="Off-Target Paralog",
                type="select",
                default="DRD2",
                options=("DRD2", "BCHE", "MAOA", "BACE2", "CHRM3", "DAT", "NONE"),
            ),
            ParamSpec(
                key="weighting",
                label="Ensemble Weighting",
                type="select",
                default="uniform",
                options=("uniform", "resolution_weighted"),
            ),
        ),
    ),
    BlockDefinition(
        block_type="reagent_library",
        name="Reagent Library",
        short_name="Reagent Library",
        category="inputs",
        python_symbol="geneuclidean.synthesis.reactions.ROBUST_MEDCHEM_REACTIONS",
        execution_tier="zero_torch",
        badge="16 SMIRKS",
        summary=(
            "Supplies a medicinal-chemistry core scaffold and curated Enamine-style coupling "
            "building blocks for makeable analog enumeration and combinatorial Thompson Sampling."
        ),
        inputs=(),
        outputs=(
            PortSpec(
                id="reagents",
                label="Reagents Pack",
                signal_type="REAGENTS",
                direction="output",
                multi=True,
                shape_hint="Core + N_B Building Blocks",
            ),
            PortSpec(
                id="core_mols",
                label="Core SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="1 Core SMILES",
            ),
        ),
        params=(
            ParamSpec(
                key="core_smiles",
                label="Core Scaffold SMILES",
                type="string",
                default="O=C(O)c1ccc2ccccc2n1",
            ),
            ParamSpec(
                key="reagent_pack",
                label="Building-Block Pack",
                type="select",
                default="cns_amines",
                options=(
                    "cns_amines",
                    "aryl_boronics",
                    "sulfonyl_chlorides",
                    "heteroaryl_halides",
                ),
            ),
            ParamSpec(
                key="max_reagents",
                label="Max Building Blocks",
                type="number",
                default=32,
                min=4,
                max=256,
                step=4,
            ),
        ),
    ),
    BlockDefinition(
        block_type="boltz2_cofold",
        name="Boltz-2 / AF3 Co-Fold",
        short_name="Boltz-2 Co-Fold",
        category="inputs",
        python_symbol="geneuclidean.interop.boltz_adapter.load_cofolded_complex",
        execution_tier="zero_torch",
        badge="mmCIF/PDB",
        summary=(
            "Parses Boltz-2, AlphaFold 3, or Chai-1 .cif/.pdb co-folded predictions, restores "
            "covalent bond orders from SMILES templates, and emits aligned pocket, pose & ligand."
        ),
        inputs=(),
        outputs=(
            PortSpec(
                id="pocket",
                label="Co-Folded Pocket",
                signal_type="POCKET_3D",
                direction="output",
                multi=True,
                shape_hint="Pocket PDB",
            ),
            PortSpec(
                id="pose",
                label="Co-Folded Pose",
                signal_type="POSE_3D",
                direction="output",
                multi=True,
                shape_hint="CoFolded Complex",
            ),
            PortSpec(
                id="ligand_mols",
                label="Ligand SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="1 Ligand SMILES",
            ),
        ),
        params=(
            ParamSpec(
                key="structure_source",
                label="Co-Fold Structure",
                type="select",
                default="ACHE_donepezil_cofold.cif",
                options=(
                    "ACHE_donepezil_cofold.cif",
                    "DRD1_ecopipam_cofold.cif",
                    "MAOB_safinamide_cofold.cif",
                    "CHRM4_xanomeline_cofold.cif",
                    "custom_path",
                ),
            ),
            ParamSpec(
                key="ligand_resname",
                label="Ligand CCD / Resname",
                type="string",
                default="LIG",
            ),
            ParamSpec(
                key="reference_smiles",
                label="Template SMILES",
                type="string",
                default="COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
            ),
            ParamSpec(
                key="pocket_cutoff",
                label="Pocket Cutoff",
                type="number",
                default=10.0,
                min=6.0,
                max=15.0,
                step=0.5,
                unit="Å",
            ),
        ),
    ),
    BlockDefinition(
        block_type="chembl_pool",
        name="ChEMBL / ZINC Pool",
        short_name="ChEMBL Pool",
        category="inputs",
        python_symbol="geneuclidean.data.splits.compute_murcko_scaffold",
        execution_tier="zero_torch",
        badge="Murcko Dedup",
        summary=(
            "Streams a Bemis–Murcko scaffold-deduplicated human ChEMBL 37 or ZINC-250k "
            "compound deck for zero-leakage virtual screening and Active Learning DMTA campaigns."
        ),
        inputs=(),
        outputs=(
            PortSpec(
                id="molecules",
                label="Scaffold Pool",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Pool[N scaffolds]",
            ),
        ),
        params=(
            ParamSpec(
                key="dataset",
                label="Bioactivity Deck",
                type="select",
                default="DRD1_chembl37_400",
                options=(
                    "DRD1_chembl37_400",
                    "ACHE_chembl37_300",
                    "MAOB_chembl37_300",
                    "BACE1_chembl37_300",
                    "SERT_chembl37_300",
                    "ZINC250k_diverse",
                ),
            ),
            ParamSpec(
                key="pool_size",
                label="Pool Size",
                type="number",
                default=400,
                min=32,
                max=5000,
                step=32,
            ),
            ParamSpec(
                key="dedup_murcko",
                label="Strict Bemis–Murcko Dedup",
                type="boolean",
                default=True,
            ),
        ),
    ),
    # =====================================================================
    # Category 2 — SE(3) / 3D Neural Encoders (5 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="painn_se3",
        name="PaiNN-SE(3)",
        short_name="PaiNN-SE(3)",
        category="encoders",
        python_symbol="geneuclidean.models.encoders.PaiNNEncoder",
        execution_tier="optional_torch",
        badge="SE(3) · 4L",
        summary=(
            "Chiral float32 SE(3)-equivariant Polarizable Atom Interaction Network "
            "(SO(3) error < 8e-6, CASF-2016 pKd r = 0.746)."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="pocket_graph",
                label="SE(3) Graph",
                signal_type="SE3_GRAPH",
                direction="output",
                multi=True,
                shape_hint="[B, N, 256] + [B, N, 64, 3]",
            ),
            PortSpec(
                id="pocket_vec",
                label="Pooled Vec",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[B, 256]",
            ),
        ),
        params=(
            ParamSpec(
                key="num_layers", label="Layers", type="number", default=4, min=2, max=8, step=1
            ),
            ParamSpec(
                key="hidden_dim",
                label="Hidden Dim",
                type="select",
                default=192,
                options=(128, 192, 256, 384),
            ),
            ParamSpec(
                key="vector_dim",
                label="Vector Dim",
                type="select",
                default=64,
                options=(32, 64, 128),
            ),
            ParamSpec(
                key="out_dim",
                label="Output Dim",
                type="select",
                default=256,
                options=(128, 256, 512),
            ),
            ParamSpec(
                key="num_rbf",
                label="Radial Basis",
                type="number",
                default=24,
                min=8,
                max=64,
                step=8,
            ),
            ParamSpec(
                key="cutoff",
                label="Edge Cutoff",
                type="number",
                default=5.5,
                min=3.5,
                max=8.0,
                step=0.5,
                unit="Å",
            ),
            ParamSpec(
                key="max_neighbors",
                label="Max Neighbors",
                type="number",
                default=24,
                min=8,
                max=48,
                step=4,
            ),
            ParamSpec(
                key="use_se3_chirality",
                label="SE(3) Chiral Cross-Product",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="egnn_3d",
        name="EGNN-3D",
        short_name="EGNN-3D",
        category="encoders",
        python_symbol="geneuclidean.models.encoders.EGNNEncoder",
        execution_tier="optional_torch",
        badge="E(3) · 4L",
        summary=(
            "E(n)-equivariant graph neural network with radial Bessel edge updates and "
            "coordinate-equivariant message passing."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="pocket_graph",
                label="SE(3) Graph",
                signal_type="SE3_GRAPH",
                direction="output",
                multi=True,
                shape_hint="[B, N, 256]",
            ),
            PortSpec(
                id="pocket_vec",
                label="Pooled Vec",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[B, 256]",
            ),
        ),
        params=(
            ParamSpec(
                key="num_layers", label="Layers", type="number", default=4, min=2, max=8, step=1
            ),
            ParamSpec(
                key="hidden_dim",
                label="Hidden Dim",
                type="select",
                default=192,
                options=(128, 192, 256),
            ),
            ParamSpec(
                key="out_dim",
                label="Output Dim",
                type="select",
                default=256,
                options=(128, 256, 512),
            ),
            ParamSpec(
                key="cutoff",
                label="Edge Cutoff",
                type="number",
                default=5.5,
                min=3.5,
                max=8.0,
                step=0.5,
                unit="Å",
            ),
            ParamSpec(
                key="max_neighbors",
                label="Max Neighbors",
                type="number",
                default=24,
                min=8,
                max=48,
                step=4,
            ),
        ),
    ),
    BlockDefinition(
        block_type="e3nn_irreps",
        name="E3NN Irreps",
        short_name="E3NN Irreps",
        category="encoders",
        python_symbol="geneuclidean.models.encoders.E3NNEncoder",
        execution_tier="optional_torch",
        badge="o3.Irreps",
        summary=(
            "Full spherical-harmonics irreducible-representation (o3.Irreps) tensor-product "
            "3D pocket encoder with IrrepsEquivariantRMSNorm."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="pocket_graph",
                label="Irreps Graph",
                signal_type="SE3_GRAPH",
                direction="output",
                multi=True,
                shape_hint="Irreps[l≤lmax] → 256D",
            ),
            PortSpec(
                id="pocket_vec",
                label="Pooled Vec",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[B, 256]",
            ),
        ),
        params=(
            ParamSpec(
                key="num_layers", label="Layers", type="number", default=4, min=2, max=6, step=1
            ),
            ParamSpec(
                key="hidden_dim",
                label="Hidden Dim",
                type="select",
                default=192,
                options=(128, 192, 256),
            ),
            ParamSpec(
                key="e3nn_lmax",
                label="Spherical Harmonic l_max",
                type="select",
                default=1,
                options=(1, 2),
            ),
            ParamSpec(
                key="use_se3_chirality",
                label="SE(3) Chiral Parity",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="cutoff",
                label="Edge Cutoff",
                type="number",
                default=5.5,
                min=3.5,
                max=8.0,
                step=0.5,
                unit="Å",
            ),
        ),
    ),
    BlockDefinition(
        block_type="drugclip_encoder",
        name="DrugCLIP Encoder",
        short_name="DrugCLIP",
        category="encoders",
        python_symbol="geneuclidean.screening.encoders.DrugCLIPLigandEncoder",
        execution_tier="optional_torch",
        badge="R@5 62.1%",
        summary=(
            "Projects ligands into GenEuclidean's 3D-pocket-aligned unit hypersphere "
            "(CASF R@5 = 62.1%, ROC-AUC = 0.937) or pure-RDKit L2-normalized ECFP4 baseline."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Molecules",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="pocket_graph",
                label="SE(3) Context",
                signal_type="SE3_GRAPH",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="embeddings",
                label="Unit Vectors",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[N, 128] float16",
            ),
        ),
        params=(
            ParamSpec(
                key="backend",
                label="Encoder Backend",
                type="select",
                default="drugclip_v2",
                options=("drugclip_v2", "ecfp4_binary"),
            ),
            ParamSpec(
                key="clip_dim",
                label="Embedding Dim",
                type="select",
                default=128,
                options=(128, 256, 1024),
            ),
            ParamSpec(
                key="batch_size",
                label="Batch Size",
                type="number",
                default=2048,
                min=64,
                max=8192,
                step=64,
            ),
        ),
    ),
    BlockDefinition(
        block_type="ibm_mmelon",
        name="IBM MMELON (84M)",
        short_name="IBM MMELON",
        category="encoders",
        python_symbol="geneuclidean.screening.encoders.IBMMMelonLigandEncoder",
        execution_tier="optional_torch",
        badge="84M · 768D",
        summary=(
            "Apache-2.0 IBM Research multi-view molecular foundation encoder (84M params, "
            "768-D unit embeddings) for hybrid 2D/SMILES + 3D screening."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Molecules",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="embeddings",
                label="768D Vectors",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[N, 768] float32",
            ),
        ),
        params=(
            ParamSpec(
                key="model_id",
                label="HF Backbone ID",
                type="string",
                default="ibm-research/biomed.sm.mv-te-84m",
            ),
            ParamSpec(
                key="batch_size",
                label="Batch Size",
                type="number",
                default=64,
                min=16,
                max=256,
                step=16,
            ),
            ParamSpec(
                key="device",
                label="Compute Device",
                type="select",
                default="auto",
                options=("auto", "mps", "cuda", "cpu"),
            ),
        ),
    ),
    # =====================================================================
    # Category 3 — De Novo Generators & Steering (4 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="safe_gpt",
        name="SAFE-GPT (87M)",
        short_name="SAFE-GPT 87M",
        category="generators",
        python_symbol="geneuclidean.sdk.designer.MoleculeDesigner",
        execution_tier="optional_torch",
        badge="87M · SAFE",
        summary=(
            "Fragment-modular SAFE chemical transformer (87.3M, 24-token Perceiver bridge, "
            "5.5× v4 Pharma-Safe, 0% DCU) supporting de novo design and 3D scaffold decoration."
        ),
        inputs=(
            PortSpec(
                id="pocket_graph",
                label="SE(3) Pocket",
                signal_type="SE3_GRAPH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="seed_mols",
                label="Seed Cores",
                signal_type="MOL_BATCH",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="feedback_scores",
                label="DMTA Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="molecules",
                label="Designed SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Batch[n SMILES]",
            ),
            PortSpec(
                id="latent",
                label="Prefix Latent",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[B, 24, 768]",
            ),
        ),
        params=(
            ParamSpec(
                key="mode",
                label="Generation Mode",
                type="select",
                default="de_novo",
                options=("de_novo", "decorate_scaffold"),
            ),
            ParamSpec(
                key="num_samples",
                label="Num Samples",
                type="number",
                default=32,
                min=4,
                max=256,
                step=4,
            ),
            ParamSpec(
                key="strategy",
                label="Decoding Strategy",
                type="select",
                default="min_p",
                options=("min_p", "top_p", "top_k", "typical", "temperature"),
            ),
            ParamSpec(
                key="cfg_scale",
                label="CFG Guidance w",
                type="number",
                default=1.6,
                min=0.5,
                max=3.0,
                step=0.05,
            ),
            ParamSpec(
                key="temperature",
                label="Temperature",
                type="number",
                default=0.95,
                min=0.5,
                max=1.4,
                step=0.05,
            ),
            ParamSpec(
                key="min_p",
                label="Min-P Cutoff",
                type="number",
                default=0.08,
                min=0.01,
                max=0.30,
                step=0.01,
            ),
            ParamSpec(
                key="num_queries",
                label="Perceiver Queries Q",
                type="number",
                default=16,
                min=4,
                max=32,
                step=4,
            ),
            ParamSpec(
                key="core_smiles_with_dummy",
                label="Core Scaffold ([*] for Decorate)",
                type="string",
                default="",
            ),
        ),
    ),
    BlockDefinition(
        block_type="chemical_llm",
        name="Chemical Foundation LLM",
        short_name="Chemical LLM",
        category="generators",
        python_symbol="geneuclidean.models.llm_bridge.PocketConditionedLLM",
        execution_tier="optional_torch",
        badge="46M–300M",
        summary=(
            "Pluggable 3D-pocket-conditioned SMILES generator supporting verified real backbones: "
            "GP-MoLFormer-Uniq, NovoMolGen-157M/300M, gpt2_zinc_87m, and v2 PocketTransformer."
        ),
        inputs=(
            PortSpec(
                id="pocket_graph",
                label="SE(3) Pocket",
                signal_type="SE3_GRAPH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="feedback_scores",
                label="DMTA Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="molecules",
                label="Designed SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Batch[n SMILES]",
            ),
            PortSpec(
                id="latent",
                label="Prefix Latent",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="[B, 24, d_llm]",
            ),
        ),
        params=(
            ParamSpec(
                key="backbone_id",
                label="Backbone",
                type="select",
                default="ibm-research/GP-MoLFormer-Uniq",
                options=(
                    "ibm-research/GP-MoLFormer-Uniq",
                    "chandar-lab/NovoMolGen_157M_SMILES_BPE",
                    "chandar-lab/NovoMolGen_300M_SMILES_BPE",
                    "entropy/gpt2_zinc_87m",
                    "v2-PocketTransformerDecoder",
                    "mock-tiny-gpt2",
                ),
            ),
            ParamSpec(
                key="num_samples",
                label="Num Samples",
                type="number",
                default=32,
                min=4,
                max=256,
                step=4,
            ),
            ParamSpec(
                key="strategy",
                label="Decoding Strategy",
                type="select",
                default="min_p",
                options=("min_p", "top_p", "top_k", "typical", "diverse_beam"),
            ),
            ParamSpec(
                key="cfg_scale",
                label="CFG Guidance w",
                type="number",
                default=1.35,
                min=0.5,
                max=3.0,
                step=0.05,
            ),
            ParamSpec(
                key="temperature",
                label="Temperature",
                type="number",
                default=0.90,
                min=0.5,
                max=1.4,
                step=0.05,
            ),
            ParamSpec(
                key="perceiver_layers",
                label="Perceiver Layers",
                type="number",
                default=2,
                min=1,
                max=4,
                step=1,
            ),
        ),
    ),
    BlockDefinition(
        block_type="pocket_ensemble_cfg",
        name="Pocket Ensemble CFG",
        short_name="Ensemble CFG",
        category="generators",
        python_symbol="geneuclidean.sampling.pocket_steering.sample_contrastive_ensemble_llm",
        execution_tier="optional_torch",
        badge="ΔPMI_3D",
        summary=(
            "Multi-conformation K-pocket logit ensembling with zero-retraining off-target "
            "repulsion and exact ΔPMI_3D specificity scoring."
        ),
        inputs=(
            PortSpec(
                id="pocket_graph",
                label="On-Target SE(3)",
                signal_type="SE3_GRAPH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="off_pocket",
                label="Off-Target Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="molecules",
                label="Steered SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Steered Batch[n]",
            ),
            PortSpec(
                id="pmi_scores",
                label="PMI_3D / ΔPMI",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="PMI_3D / ΔPMI",
            ),
        ),
        params=(
            ParamSpec(
                key="backbone_id",
                label="Backbone",
                type="select",
                default="datamol-io/safe-gpt",
                options=(
                    "datamol-io/safe-gpt",
                    "ibm-research/GP-MoLFormer-Uniq",
                    "chandar-lab/NovoMolGen_157M_SMILES_BPE",
                ),
            ),
            ParamSpec(
                key="w_on",
                label="On-Target Guidance w_on",
                type="number",
                default=1.6,
                min=0.5,
                max=3.0,
                step=0.05,
            ),
            ParamSpec(
                key="w_off",
                label="Off-Target Repulsion w_off",
                type="number",
                default=0.65,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ParamSpec(
                key="num_samples",
                label="Num Samples",
                type="number",
                default=32,
                min=4,
                max=128,
                step=4,
            ),
            ParamSpec(
                key="compute_pmi_3d",
                label="Compute ΔPMI_3D Selectivity",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="selectivity_mtl_aligner",
        name="Selectivity & MTL Aligner",
        short_name="Selectivity MTL",
        category="generators",
        python_symbol="geneuclidean.training.mtl_intervention.GradientConflictMediator",
        execution_tier="optional_torch",
        badge="AIM / GDPO",
        summary=(
            "Fine-tunes contrastive paralog selectivity margins (DRD1 vs DRD2 AUROC 0.417→0.735) "
            "and resolves multi-task gradient conflicts via AIM Policy / PCGrad / 4-Ch GDPO."
        ),
        inputs=(
            PortSpec(
                id="pocket_graph",
                label="On-Target SE(3)",
                signal_type="SE3_GRAPH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="off_pocket",
                label="Off-Target Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="feedback_scores",
                label="Oracle Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="aligned_vec",
                label="Aligned Head",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="Tuned Selectivity Head",
            ),
            PortSpec(
                id="diagnostics",
                label="MTL Diagnostics",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="Grad Conflict + Margin",
            ),
        ),
        params=(
            ParamSpec(
                key="margin",
                label="Paralog Margin m",
                type="number",
                default=0.15,
                min=0.05,
                max=0.50,
                step=0.05,
            ),
            ParamSpec(
                key="selectivity_weight",
                label="Selectivity Weight",
                type="number",
                default=0.50,
                min=0.10,
                max=2.00,
                step=0.05,
            ),
            ParamSpec(
                key="mtl_strategy",
                label="MTL Strategy",
                type="select",
                default="aim_policy",
                options=("aim_policy", "pcgrad", "scalarized"),
            ),
            ParamSpec(
                key="gdpo_beta",
                label="GDPO KL β",
                type="number",
                default=0.04,
                min=0.01,
                max=0.50,
                step=0.01,
            ),
            ParamSpec(
                key="steps",
                label="Alignment Steps",
                type="number",
                default=30,
                min=10,
                max=200,
                step=10,
            ),
        ),
    ),
    # =====================================================================
    # Category 4 — Virtual Screening & Retrieval (4 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="ligand_vector_index",
        name="Ligand Vector Index",
        short_name="Vector Index",
        category="screening",
        python_symbol="geneuclidean.screening.screener.LigandVectorIndex",
        execution_tier="zero_torch",
        badge="Float16 BLAS",
        summary=(
            "Streaming float16 L2-normalized vector index supporting chunked BLAS cosine "
            "retrieval over Parquet/CSV/SMILES catalogs in bounded memory."
        ),
        inputs=(
            PortSpec(
                id="embeddings",
                label="Embeddings",
                signal_type="LATENT_VEC",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="molecules",
                label="Catalog SMILES",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="index_vecs",
                label="Vector Index",
                signal_type="LATENT_VEC",
                direction="output",
                multi=True,
                shape_hint="Float16 Index[N×D]",
            ),
            PortSpec(
                id="pool_mols",
                label="Indexed Pool",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Pool[N]",
            ),
        ),
        params=(
            ParamSpec(
                key="chunk_size",
                label="Chunk Size",
                type="number",
                default=65536,
                min=1024,
                max=262144,
                step=1024,
            ),
            ParamSpec(
                key="dtype",
                label="Storage Dtype",
                type="select",
                default="float16",
                options=("float16", "float32"),
            ),
            ParamSpec(
                key="mmap",
                label="Memory-Mapped Index",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="pocket_screener",
        name="Contrastive Pocket Screener",
        short_name="Pocket Screener",
        category="screening",
        python_symbol="geneuclidean.screening.screener.PocketScreener",
        execution_tier="optional_torch",
        badge="0.02 ms/q",
        summary=(
            "Millisecond multi-conformation 3D dense retrieval with optional off-target "
            "vector repulsion s(m) = v_on^T u_m - λ v_off^T u_m."
        ),
        inputs=(
            PortSpec(
                id="on_pockets",
                label="On-Target Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="off_pocket",
                label="Off-Target Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="index_vecs",
                label="Vector Index",
                signal_type="LATENT_VEC",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="pool_mols",
                label="Indexed Pool",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="hits",
                label="Top-K Hits",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Top-K ScreeningHits",
            ),
            PortSpec(
                id="scores",
                label="CLIP + ΔpKd",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="CLIP + ΔpKd",
            ),
        ),
        params=(
            ParamSpec(
                key="off_weight",
                label="Off-Target Repulsion λ",
                type="number",
                default=0.50,
                min=0.0,
                max=1.5,
                step=0.05,
            ),
            ParamSpec(
                key="top_k",
                label="Top-K Hits",
                type="number",
                default=50,
                min=5,
                max=500,
                step=5,
            ),
            ParamSpec(
                key="rerank_pkd",
                label="Stage-2 Neural pKd Rerank",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="triage_filter",
                label="MedChem Hygiene Gate",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="neural_pkd_reranker",
        name="Neural pKd & MaxMin Reranker",
        short_name="pKd + MaxMin",
        category="screening",
        python_symbol="geneuclidean.sampling.rerank.rerank_and_select",
        execution_tier="optional_torch",
        badge="MaxMin Div",
        summary=(
            "Reranks candidate molecules by neural 3D pKd / MPO desirability and enforces "
            "Bemis–Murcko chemical diversity via greedy MaxMin Tanimoto selection."
        ),
        inputs=(
            PortSpec(
                id="pocket_graph",
                label="SE(3) Pocket",
                signal_type="SE3_GRAPH",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="molecules",
                label="Candidates",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="scores",
                label="Prior Scores",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="ranked_mols",
                label="Diverse Top-K",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Diverse Top-K",
            ),
            PortSpec(
                id="scores",
                label="pKd + MPO",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="pKd + MPO",
            ),
        ),
        params=(
            ParamSpec(
                key="top_k", label="Top-K", type="number", default=16, min=4, max=128, step=4
            ),
            ParamSpec(
                key="min_tanimoto_dist",
                label="Min Tanimoto Dist",
                type="number",
                default=0.35,
                min=0.10,
                max=0.70,
                step=0.05,
            ),
            ParamSpec(
                key="quality_weight",
                label="Quality Weight",
                type="number",
                default=0.70,
                min=0.20,
                max=1.00,
                step=0.05,
            ),
        ),
    ),
    BlockDefinition(
        block_type="catalog_grounding",
        name="Catalog Grounding",
        short_name="Catalog Ground",
        category="screening",
        python_symbol="geneuclidean.screening.screener.PocketScreener.ground_to_catalog",
        execution_tier="zero_torch",
        badge="Buyable NN",
        summary=(
            "Maps de novo generated molecules onto nearest buyable/make-on-demand catalog "
            "analogs ranked jointly by 3D contrastive pocket score (α_3D) and 2D ECFP4 Tanimoto."
        ),
        inputs=(
            PortSpec(
                id="query_mols",
                label="Query SMILES",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="index_vecs",
                label="Catalog Index",
                signal_type="LATENT_VEC",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="on_pockets",
                label="Target Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="grounded_mols",
                label="Buyable Analogs",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Buyable Analogs",
            ),
            PortSpec(
                id="scores",
                label="Tanimoto + 3D",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="Tanimoto + 3D Score",
            ),
        ),
        params=(
            ParamSpec(
                key="top_k_per_query",
                label="Top-K / Query",
                type="number",
                default=5,
                min=1,
                max=20,
                step=1,
            ),
            ParamSpec(
                key="alpha_3d",
                label="3D vs 2D Blend α_3D",
                type="number",
                default=0.50,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParamSpec(
                key="off_weight",
                label="Off-Target Weight",
                type="number",
                default=0.50,
                min=0.0,
                max=1.5,
                step=0.05,
            ),
        ),
    ),
    # =====================================================================
    # Category 5 — Active Learning DMTA (6 blocks, v5.1 + fallback)
    # =====================================================================
    BlockDefinition(
        block_type="tanimoto_gp",
        name="Tanimoto-Kernel GP",
        short_name="Tanimoto GP",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.surrogates.TanimotoGP",
        execution_tier="v51_al",
        badge="NEW v5.1",
        summary=(
            "Exact Gaussian Process with analytical positive-semidefinite Tanimoto / MinMax "
            "fingerprint kernel and Cholesky decomposition for closed-form posterior mean & std."
        ),
        inputs=(
            PortSpec(
                id="pool_mols",
                label="Candidate Pool",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="observations",
                label="Assay Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="posterior",
                label="GP Posterior",
                signal_type="UNCERTAINTY_DIST",
                direction="output",
                multi=True,
                shape_hint="GP(μ, σ²_exact)",
            ),
        ),
        params=(
            ParamSpec(
                key="kernel",
                label="Kernel",
                type="select",
                default="tanimoto",
                options=("tanimoto", "minmax_count"),
            ),
            ParamSpec(
                key="fp_bits",
                label="ECFP4 Bits",
                type="select",
                default=2048,
                options=(1024, 2048),
            ),
            ParamSpec(
                key="noise_var",
                label="Observation Noise σ²_n",
                type="number",
                default=0.05,
                min=0.001,
                max=1.0,
                step=0.01,
            ),
            ParamSpec(
                key="signal_var",
                label="Kernel Variance σ²_f",
                type="number",
                default=1.0,
                min=0.1,
                max=5.0,
                step=0.1,
            ),
            ParamSpec(
                key="cold_start",
                label="Cold-Start Seeding",
                type="select",
                default="butina_medoids",
                options=("butina_medoids", "maxmin", "random"),
            ),
        ),
    ),
    BlockDefinition(
        block_type="ensemble_surrogate",
        name="Ensemble Surrogate",
        short_name="RF+HGB Surrogate",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.surrogates.EnsembleSurrogate",
        execution_tier="v51_al",
        badge="NEW v5.1",
        summary=(
            "Bootstrap RandomForest + HistGradientBoosting ensemble surrogate with explicit "
            "epistemic vs. aleatoric uncertainty decomposition over ECFP4 + 2D descriptors."
        ),
        inputs=(
            PortSpec(
                id="pool_mols",
                label="Candidate Pool",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="observations",
                label="Assay Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="posterior",
                label="Ensemble UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="output",
                multi=True,
                shape_hint="Ensemble(μ, σ_ep, σ_al)",
            ),
        ),
        params=(
            ParamSpec(
                key="n_estimators",
                label="Estimators",
                type="number",
                default=64,
                min=16,
                max=256,
                step=16,
            ),
            ParamSpec(
                key="fp_bits",
                label="ECFP4 Bits",
                type="select",
                default=1024,
                options=(1024, 2048),
            ),
            ParamSpec(
                key="include_2d_desc",
                label="Append 12 RDKit Descriptors",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="conformal_calibrator",
        name="Split-Conformal UQ",
        short_name="Conformal UQ",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.surrogates.ConformalSurrogate",
        execution_tier="v51_al",
        badge="1-α Cov",
        summary=(
            "Wraps any surrogate posterior in distribution-free finite-sample Split-Conformal "
            "intervals supporting homoscedastic, normalized, and weighted covariate-shift modes."
        ),
        inputs=(
            PortSpec(
                id="posterior",
                label="Raw Posterior",
                signal_type="UNCERTAINTY_DIST",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="calibration_scores",
                label="Cal Labels",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="calibrated_uq",
                label="Calibrated UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="output",
                multi=True,
                shape_hint="Conformal[lo, hi] @ (1-α)",
            ),
        ),
        params=(
            ParamSpec(
                key="mode",
                label="Conformal Mode",
                type="select",
                default="normalized",
                options=("normalized", "homoscedastic", "weighted_covariate_shift"),
            ),
            ParamSpec(
                key="alpha",
                label="Miscoverage α (90% = 0.10)",
                type="number",
                default=0.10,
                min=0.02,
                max=0.30,
                step=0.02,
            ),
            ParamSpec(
                key="scaffold_grouped",
                label="Bemis–Murcko Cal Split",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="batch_acquisition",
        name="Batch Acquisition",
        short_name="Batch Acq",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.acquisitions.ACQUISITION_REGISTRY",
        execution_tier="v51_al",
        badge="12 Strategies",
        summary=(
            "Selects diverse, uncertainty-aware batches across 12 acquisition strategies "
            "(KrigingBeliever, DPP-UCB, MaxMin-UCB, qEHVI, ParEGO, Log-EI, Constrained-EI) "
            "with async in-flight pending-batch masking."
        ),
        inputs=(
            PortSpec(
                id="posterior",
                label="Posterior UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="pool_mols",
                label="Candidate Pool",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="constraints",
                label="Feasibility Constraints",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                multi=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="selected_batch",
                label="Query Batch",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Query Batch[B]",
            ),
            PortSpec(
                id="acq_scores",
                label="Acq Scores",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="Acq + Pareto HV",
            ),
        ),
        params=(
            ParamSpec(
                key="strategy",
                label="Acquisition Strategy",
                type="select",
                default="kriging_believer",
                options=(
                    "kriging_believer",
                    "dpp_ucb",
                    "maxmin_ucb",
                    "qehvi",
                    "pareto_scalarized_ucb",
                    "log_ei",
                    "constrained_ei",
                    "avoidome_feasibility_ucb",
                    "ucb",
                    "ei",
                    "pi",
                    "thompson",
                ),
            ),
            ParamSpec(
                key="batch_size",
                label="Batch Size B",
                type="number",
                default=12,
                min=2,
                max=64,
                step=2,
            ),
            ParamSpec(
                key="ucb_beta",
                label="Exploration β",
                type="number",
                default=1.5,
                min=0.1,
                max=4.0,
                step=0.1,
            ),
            ParamSpec(
                key="diversity_weight",
                label="Diversity Weight",
                type="number",
                default=0.30,
                min=0.0,
                max=0.9,
                step=0.05,
            ),
            ParamSpec(
                key="mask_pending_async",
                label="Async In-Flight Batch Mask",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="multifidelity_router",
        name="Multi-Fidelity Router",
        short_name="MF-DMTA Router",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.multifidelity.MultiFidelityDMTA",
        execution_tier="v51_al",
        badge="ρ≥0.30 Guard",
        summary=(
            "Cost-aware Kennedy–O'Hagan residual GP router across Tier-0 surrogate, Tier-1 "
            "Vina/PoseRealityAuditor, and Tier-2 RBFE/wet-lab with negative-transfer guardrail."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Query Batch",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="posterior",
                label="Posterior UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="tier1_batch",
                label="Tier-1 Vina",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Tier-1 Vina/Audit",
            ),
            PortSpec(
                id="tier2_batch",
                label="Tier-2 RBFE",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Tier-2 RBFE/Wet-Lab",
            ),
            PortSpec(
                id="routing_report",
                label="Routing Ledger",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="Cost/VoI Ledger",
            ),
        ),
        params=(
            ParamSpec(
                key="tier1_budget",
                label="Tier-1 Vina Quota",
                type="number",
                default=12,
                min=2,
                max=64,
                step=2,
            ),
            ParamSpec(
                key="tier2_budget",
                label="Tier-2 RBFE/Assay Quota",
                type="number",
                default=4,
                min=1,
                max=16,
                step=1,
            ),
            ParamSpec(
                key="cost_ratio_t2_t1",
                label="Cost Ratio (Tier2 / Tier1)",
                type="number",
                default=20.0,
                min=2.0,
                max=100.0,
                step=2.0,
            ),
            ParamSpec(
                key="min_spearman_rho",
                label="Negative-Transfer Guardrail ρ",
                type="number",
                default=0.30,
                min=0.0,
                max=0.80,
                step=0.05,
            ),
        ),
    ),
    BlockDefinition(
        block_type="combinatorial_ts",
        name="Combinatorial Thompson Sampler",
        short_name="Combinatorial TS",
        category="active_learning",
        python_symbol="geneuclidean.active_learning.combinatorial_ts.CombinatorialThompsonSampler",
        execution_tier="v51_al",
        badge="O(N_A+N_B)",
        summary=(
            "Explores N_A × N_B reaction-grounded Enamine-style virtual spaces in O(N_A + N_B) "
            "oracle calls using reagent-level Bayesian Thompson posteriors with MedChem filtering."
        ),
        inputs=(
            PortSpec(
                id="reagents",
                label="Reagents Pack",
                signal_type="REAGENTS",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="feedback_scores",
                label="Assay Feedback",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                accepts_feedback=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="sampled_mols",
                label="Sampled Products",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Sampled Products",
            ),
            PortSpec(
                id="posterior",
                label="Reagent Posteriors",
                signal_type="UNCERTAINTY_DIST",
                direction="output",
                multi=True,
                shape_hint="Reagent Posteriors",
            ),
        ),
        params=(
            ParamSpec(
                key="reaction_name",
                label="Reaction SMIRKS",
                type="select",
                default="amide_coupling_acid_amine",
                options=_REACTION_OPTIONS,
            ),
            ParamSpec(
                key="batch_size",
                label="Batch Size",
                type="number",
                default=16,
                min=4,
                max=64,
                step=4,
            ),
            ParamSpec(
                key="warmup_per_reagent",
                label="Warmup / Reagent",
                type="number",
                default=2,
                min=1,
                max=5,
                step=1,
            ),
            ParamSpec(
                key="thermal_scale",
                label="Exploration Temp τ",
                type="number",
                default=1.0,
                min=0.2,
                max=3.0,
                step=0.1,
            ),
            ParamSpec(
                key="triage_filter",
                label="MedChem Hygiene Filter",
                type="boolean",
                default=True,
            ),
        ),
    ),
    # =====================================================================
    # Category 6 — 3D Physics & 7-Axis Pose Reality (4 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="protonation_ph74",
        name="pH 7.4 Protonation & Tautomer",
        short_name="pH 7.4 Microstate",
        category="physics_audit",
        python_symbol="geneuclidean.evaluation.protonation.protonate_smiles_ph74",
        execution_tier="zero_torch",
        badge="pH 7.4",
        summary=(
            "Ionizes basic amines and carboxylic acids at physiological pH 7.4 (essential for "
            "GPCR Asp3.32 salt bridges) and audits minor tautomer stability gaps."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Input SMILES",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="protonated_mols",
                label="pH 7.4 SMILES",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="pH 7.4 SMILES",
            ),
            PortSpec(
                id="microstate_report",
                label="Microstate Audit",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="Charge + Tautomer Audit",
            ),
        ),
        params=(
            ParamSpec(
                key="audit_tautomers",
                label="Audit Tautomer Microstates",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="max_tautomers",
                label="Max Tautomers",
                type="number",
                default=16,
                min=4,
                max=32,
                step=4,
            ),
            ParamSpec(
                key="tautomer_gap_threshold",
                label="Tautomer Gap Threshold",
                type="number",
                default=2.0,
                min=0.5,
                max=5.0,
                step=0.25,
            ),
            ParamSpec(
                key="strict_input_charge",
                label="Strict Input Charge",
                type="boolean",
                default=False,
            ),
        ),
    ),
    BlockDefinition(
        block_type="autodock_vina",
        name="AutoDock Vina 3D",
        short_name="AutoDock Vina",
        category="physics_audit",
        python_symbol="geneuclidean.evaluation.docking_pipeline.dock_ligand_to_pocket",
        execution_tier="zero_torch",
        badge="Vina 1.2.7",
        summary=(
            "Generates ring-diverse ETKDGv3 conformers, docks into the orthosteric box with "
            "AutoDock Vina v1.2.7, and profiles 6-channel 3D PLIF & canonical anchors."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="molecules",
                label="Ligands",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="poses",
                label="Docked Poses",
                signal_type="POSE_3D",
                direction="output",
                multi=True,
                shape_hint="3D Docked SDFs",
            ),
            PortSpec(
                id="scores",
                label="Vina ΔG + PLIF",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="Vina ΔG, LE, PLIF",
            ),
        ),
        params=(
            ParamSpec(
                key="exhaustiveness",
                label="Exhaustiveness",
                type="number",
                default=8,
                min=2,
                max=32,
                step=2,
            ),
            ParamSpec(
                key="num_modes",
                label="Binding Modes",
                type="number",
                default=9,
                min=1,
                max=20,
                step=1,
            ),
            ParamSpec(
                key="box_mode",
                label="Orthosteric Box",
                type="select",
                default="tight_per_axis",
                options=("tight_per_axis", "cubic_standard"),
            ),
            ParamSpec(
                key="max_ring_confs",
                label="Max Ring Confs",
                type="number",
                default=4,
                min=1,
                max=8,
                step=1,
            ),
            ParamSpec(
                key="protonate_ph74",
                label="Protonate pH 7.4",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="ring_torsion_strain",
        name="Cremer–Pople & Torsion Strain",
        short_name="Ring & Torsion",
        category="physics_audit",
        python_symbol="geneuclidean.evaluation.ring_strain.analyze_ring_strain",
        execution_tier="zero_torch",
        badge="Beyond-PB",
        summary=(
            "Catches the 39.3% of PoseBusters-passing poses hiding high-energy twist_boat "
            "saturated rings (Cremer–Pople MMFF94s) or strained conjugated dihedrals."
        ),
        inputs=(
            PortSpec(
                id="poses",
                label="3D Poses",
                signal_type="POSE_3D",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="clean_poses",
                label="Low-Strain Poses",
                signal_type="POSE_3D",
                direction="output",
                multi=True,
                shape_hint="Low-Strain Poses",
            ),
            PortSpec(
                id="strain_scores",
                label="Strain kcal/mol",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="Ring + Torsion kcal/mol",
            ),
        ),
        params=(
            ParamSpec(
                key="ring_strain_threshold_kcal",
                label="Ring Strain Max",
                type="number",
                default=3.0,
                min=1.0,
                max=8.0,
                step=0.5,
                unit="kcal/mol",
            ),
            ParamSpec(
                key="ring_n_ref_confs",
                label="Ring Ref Confs",
                type="number",
                default=24,
                min=8,
                max=48,
                step=8,
            ),
            ParamSpec(
                key="torsion_barrier_threshold_kcal",
                label="Torsion Barrier Max",
                type="number",
                default=2.5,
                min=1.0,
                max=6.0,
                step=0.25,
                unit="kcal/mol",
            ),
            ParamSpec(
                key="torsion_max_red",
                label="Max Red Torsions",
                type="number",
                default=0,
                min=0,
                max=3,
                step=1,
            ),
            ParamSpec(
                key="torsion_backend",
                label="Torsion Backend",
                type="select",
                default="smarts_mmff94s",
                options=("smarts_mmff94s", "roche_xml_plugin"),
            ),
        ),
    ),
    BlockDefinition(
        block_type="pose_reality_auditor",
        name="7-Axis PoseRealityAuditor",
        short_name="7-Axis Auditor",
        category="physics_audit",
        python_symbol="geneuclidean.audit.pose_auditor.PoseRealityAuditor",
        execution_tier="zero_torch",
        badge="7-Axis Physics",
        summary=(
            "Unified Beyond-PoseBusters 7-axis physical & MedChem auditor: PoseBusters clashes, "
            "Cremer–Pople ring strain, torsion barriers, pH 7.4 microstates, PLIF, hygiene & MPO."
        ),
        inputs=(
            PortSpec(
                id="poses",
                label="3D Poses",
                signal_type="POSE_3D",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="pocket",
                label="Pocket PDB",
                signal_type="POCKET_3D",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="verified_poses",
                label="Verified Poses",
                signal_type="POSE_3D",
                direction="output",
                multi=True,
                shape_hint="7-Axis Verified Poses",
            ),
            PortSpec(
                id="reality_scores",
                label="Reality Scores",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="reality_score [0–1]",
            ),
            PortSpec(
                id="audit_report",
                label="Audit Report",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="7-Axis PoseAuditReport",
            ),
        ),
        params=(
            ParamSpec(
                key="property_profile",
                label="Property Profile",
                type="select",
                default="cns_mpo_v2",
                options=_PROPERTY_PROFILE_OPTIONS,
            ),
            ParamSpec(
                key="ring_strain_threshold_kcal",
                label="Ring Strain Max",
                type="number",
                default=3.0,
                min=1.0,
                max=6.0,
                step=0.5,
                unit="kcal/mol",
            ),
            ParamSpec(
                key="torsion_barrier_threshold_kcal",
                label="Torsion Barrier Max",
                type="number",
                default=2.5,
                min=1.0,
                max=6.0,
                step=0.25,
                unit="kcal/mol",
            ),
            ParamSpec(
                key="torsion_max_red",
                label="Max Red Torsions",
                type="number",
                default=0,
                min=0,
                max=2,
                step=1,
            ),
            ParamSpec(
                key="strict_input_charge",
                label="Strict Input Charge",
                type="boolean",
                default=False,
            ),
        ),
    ),
    # =====================================================================
    # Category 7 — Lead-Opt, ADMET & Synthesis (4 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="reaction_enumerator",
        name="16-Reaction Makeable & Retro",
        short_name="Makeable Rxns",
        category="leadopt_admet",
        python_symbol="geneuclidean.synthesis.reactions.enumerate_makeable_analogs",
        execution_tier="zero_torch",
        badge="16 Rxns + Retro",
        summary=(
            "Enumerates 1-step parallelizable MedChem analogs across 16 Hartenfeller / Enamine "
            "REAL SMIRKS couplings with full reactant provenance and 1-step retrosynthesis."
        ),
        inputs=(
            PortSpec(
                id="reagents",
                label="Reagents Pack",
                signal_type="REAGENTS",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="molecules",
                label="Molecules (Retro)",
                signal_type="MOL_BATCH",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="makeable_mols",
                label="Makeable Analogs",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="MakeableCandidates",
            ),
            PortSpec(
                id="synth_report",
                label="Synthesis Report",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="SMIRKS + Retro Routes",
            ),
        ),
        params=(
            ParamSpec(
                key="mode",
                label="Synthesis Mode",
                type="select",
                default="enumerate_and_retro",
                options=("enumerate_and_retro", "enumerate_only", "retrosynthesis_only"),
            ),
            ParamSpec(
                key="reaction_family",
                label="Reaction Family",
                type="select",
                default="all_16",
                options=(
                    "all_16",
                    "amide_coupling_acid_amine",
                    "sulfonamide_coupling",
                    "reductive_amination_aldehyde",
                    "suzuki_miyaura_biaryl",
                    "buchwald_hartwig_amination",
                    "snar_heteroaryl_fluoride",
                    "cuaac_triazole_click",
                ),
            ),
            ParamSpec(
                key="max_products",
                label="Max Products",
                type="number",
                default=64,
                min=8,
                max=500,
                step=8,
            ),
            ParamSpec(
                key="triage_filter",
                label="Quarantine DCU/PAINS",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="leadopt_mmp_sar",
        name="Avoid-ome MMP & R-Group Cliffs",
        short_name="Avoid-ome MMP",
        category="leadopt_admet",
        python_symbol="geneuclidean.leadopt.mmp.AvoidomeMMPEngine",
        execution_tier="zero_torch",
        badge="15 Bioisosteres",
        summary=(
            "Runs Roche mmpdb-inspired 1-cut bioisostere rescues (15 curated hERG/MDR1/5-HT2B "
            "transforms) and MCS R1×R2 R-group activity-cliff decomposition."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Parent Series",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="scores",
                label="Activity Values",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="rescued_mols",
                label="Rescued Analogs",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Ranked MMP Rescues",
            ),
            PortSpec(
                id="mmp_scores",
                label="ΔhERG / ΔMPO",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="ΔhERG, ΔMPO, Rescue",
            ),
            PortSpec(
                id="sar_report",
                label="R-Group SAR",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="R-Group Table + Cliffs",
            ),
        ),
        params=(
            ParamSpec(
                key="mode",
                label="Lead-Opt Mode",
                type="select",
                default="mmp_and_cliffs",
                options=("mmp_and_cliffs", "mmp_rescue_only", "rgroup_cliffs_only"),
            ),
            ParamSpec(
                key="property_profile",
                label="Property Profile",
                type="select",
                default="cns_mpo_v2",
                options=_PROPERTY_PROFILE_OPTIONS,
            ),
            ParamSpec(
                key="max_proposals",
                label="Max Proposals",
                type="number",
                default=20,
                min=5,
                max=50,
                step=5,
            ),
            ParamSpec(
                key="cliff_threshold",
                label="Activity Cliff Δ",
                type="number",
                default=1.0,
                min=0.3,
                max=3.0,
                step=0.1,
            ),
            ParamSpec(
                key="triage_filter",
                label="MedChem Hygiene Gate",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="cns_admet_suite",
        name="12-Ep Avoid-ome & Property Profile",
        short_name="Avoid-ome 12-Ep",
        category="leadopt_admet",
        python_symbol="geneuclidean.cns.admet.CNSAdmetSuite",
        execution_tier="zero_torch",
        badge="12-Ep Safety",
        summary=(
            "Scores all 12 industrial scaffold-CV Avoid-ome/DMPK endpoints (BBB, P-gp, hERG, "
            "MDR1, HLM, CYP2D6/3A4, 5-HT2B, H1, DILI, AMES, M3), triage, and property profiles."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Candidates",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
                multi=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="safe_mols",
                label="Safety-Gated Leads",
                signal_type="MOL_BATCH",
                direction="output",
                multi=True,
                shape_hint="Safety-Gated Leads",
            ),
            PortSpec(
                id="admet_scores",
                label="Avoid-ome + MPO",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="12-Ep Avoid-ome + MPO",
            ),
        ),
        params=(
            ParamSpec(
                key="property_profile",
                label="Property Profile",
                type="select",
                default="cns_mpo_v2",
                options=_PROPERTY_PROFILE_OPTIONS,
            ),
            ParamSpec(
                key="admet_profile",
                label="ADMET Suite",
                type="select",
                default="v4_12ep",
                options=("v4_12ep", "v3_core3", "profile_only"),
            ),
            ParamSpec(
                key="min_mpo_v2",
                label="Min CNS MPO v2",
                type="number",
                default=4.0,
                min=2.0,
                max=5.8,
                step=0.2,
            ),
            ParamSpec(
                key="max_herg_prob",
                label="Max P(hERG)",
                type="number",
                default=0.30,
                min=0.05,
                max=0.80,
                step=0.05,
            ),
            ParamSpec(
                key="require_triage_pass",
                label="Enforce PAINS/DCU Quarantine",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="conformal_dmpk",
        name="Conformal DMPK Regressor",
        short_name="Conformal DMPK",
        category="leadopt_admet",
        python_symbol="geneuclidean.cns.dmpk_regression.ConformalDMPKRegressor",
        execution_tier="zero_torch",
        badge="90% Cov UQ",
        summary=(
            "OpenADMET-ExpansionRx & Biogen continuous pharmacokinetic regression (LogD, "
            "KSOL, HLM CLint, Solubility) with distribution-free Split-Conformal 90% intervals."
        ),
        inputs=(
            PortSpec(
                id="molecules",
                label="Molecules",
                signal_type="MOL_BATCH",
                direction="input",
                required=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="dmpk_intervals",
                label="Conformal [lo, hi]",
                signal_type="UNCERTAINTY_DIST",
                direction="output",
                multi=True,
                shape_hint="ŷ ± q̂ [lo, hi]",
            ),
            PortSpec(
                id="dmpk_scores",
                label="DMPK Estimates",
                signal_type="ORACLE_SCORES",
                direction="output",
                multi=True,
                shape_hint="LogD / KSOL / HLM",
            ),
        ),
        params=(
            ParamSpec(
                key="endpoint",
                label="DMPK Endpoint",
                type="select",
                default="OpenADMET_LogD",
                options=(
                    "OpenADMET_LogD",
                    "OpenADMET_log10_KSOL",
                    "OpenADMET_log10_HLM_CLint",
                    "Biogen_log_HLM_CLint",
                    "Biogen_log_Solubility_pH68",
                ),
            ),
            ParamSpec(
                key="alpha",
                label="Miscoverage α",
                type="number",
                default=0.10,
                min=0.05,
                max=0.25,
                step=0.05,
            ),
            ParamSpec(
                key="n_estimators",
                label="Estimators",
                type="number",
                default=80,
                min=20,
                max=160,
                step=20,
            ),
        ),
    ),
    # =====================================================================
    # Category 8 — Outputs & Open-Pharma Export (4 blocks)
    # =====================================================================
    BlockDefinition(
        block_type="openfe_rbfe_exporter",
        name="OpenFE RBFE Exporter",
        short_name="OpenFE RBFE",
        category="outputs",
        python_symbol="geneuclidean.interop.openfe_export.export_openfe_rbfe_network",
        execution_tier="zero_torch",
        badge="Δq = 0 Lomap",
        summary=(
            "Exports a charge-consistent (Δq = 0), ring-strain-clean, MCS-aligned Lomap "
            "RBFE perturbation graph with cycle-closure edges and runnable OpenFE driver."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="Receptor Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="molecules",
                label="Congeneric SMILES",
                signal_type="MOL_BATCH",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="poses",
                label="3D Poses",
                signal_type="POSE_3D",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="rbfe_network",
                label="RBFE Network",
                signal_type="RBFE_GRAPH",
                direction="output",
                multi=True,
                shape_hint="Lomap Graph + OpenFE Driver",
            ),
        ),
        params=(
            ParamSpec(
                key="require_same_charge",
                label="Enforce Δq = 0 Guard",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="filter_ring_strain",
                label="RingStrainFilter Pre-Check",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="max_ring_strain_kcal",
                label="Max Ring Strain",
                type="number",
                default=3.0,
                min=1.0,
                max=6.0,
                step=0.5,
                unit="kcal/mol",
            ),
            ParamSpec(
                key="add_cycle_closures",
                label="Add Hysteresis Cycle Closures",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="out_dir",
                label="Export Directory",
                type="string",
                default="results/openfe_circuit_export",
            ),
        ),
    ),
    BlockDefinition(
        block_type="studio_3d_inspector",
        name="3D Studio Inspector",
        short_name="3D Inspector",
        category="outputs",
        python_symbol="geneuclidean.sdk.viz.visualize_complex",
        execution_tier="zero_torch",
        badge="3Dmol WebGL",
        summary=(
            "Renders interactive 3Dmol.js WebGL protein-ligand complexes with 6-channel PLIF "
            "contacts and residue attention heatmaps, deep-linking straight to the 3D Studio tab."
        ),
        inputs=(
            PortSpec(
                id="pocket",
                label="3D Pocket",
                signal_type="POCKET_3D",
                direction="input",
                required=True,
            ),
            PortSpec(
                id="poses",
                label="3D Poses",
                signal_type="POSE_3D",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="scores",
                label="KPI Scores",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="audit_report",
                label="Audit Report",
                signal_type="PROVENANCE_REPORT",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="viz_manifest",
                label="3D Viz Spec",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="3D WebGL + PNG Spec",
            ),
        ),
        params=(
            ParamSpec(
                key="theme",
                label="Render Theme",
                type="select",
                default="dark",
                options=("dark", "light"),
            ),
            ParamSpec(
                key="show_native_ghost",
                label="Overlay Native Co-Crystal",
                type="boolean",
                default=True,
            ),
            ParamSpec(
                key="highlight_anchors",
                label="Highlight Canonical Anchors",
                type="boolean",
                default=True,
            ),
        ),
    ),
    BlockDefinition(
        block_type="mcp_tool_export",
        name="Agentic MCP Tool Server",
        short_name="MCP Tool Export",
        category="outputs",
        python_symbol="geneuclidean.mcp_server.invoke_dmta_tool",
        execution_tier="zero_torch",
        badge="6 MCP Tools",
        summary=(
            "Packages pipeline endpoints as deterministic, SHA-256 provenance-stamped Model "
            "Context Protocol (MCP) tools for autonomous pharma DMTA agents."
        ),
        inputs=(
            PortSpec(
                id="audit_report",
                label="Reports",
                signal_type="PROVENANCE_REPORT",
                direction="input",
                required=False,
                multi=True,
            ),
            PortSpec(
                id="uncertainty",
                label="Conformal UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="molecules",
                label="Molecules",
                signal_type="MOL_BATCH",
                direction="input",
                required=False,
            ),
        ),
        outputs=(
            PortSpec(
                id="mcp_manifest",
                label="MCP Manifest",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="6 Typed MCP Tools + SHA-256",
            ),
        ),
        params=(
            ParamSpec(
                key="tool_name",
                label="DMTA Tool",
                type="select",
                default="audit_complex",
                options=(
                    "score_molecules",
                    "evaluate_property_profile",
                    "audit_complex",
                    "enumerate_makeable_analogs",
                    "propose_mmp_rescues",
                    "match_retrosynthesis",
                ),
            ),
            ParamSpec(
                key="transport",
                label="Transport",
                type="select",
                default="python_direct",
                options=("python_direct", "mcp_stdio"),
            ),
        ),
    ),
    BlockDefinition(
        block_type="dmta_ledger",
        name="DMTA Ledger & Stopping Advisor",
        short_name="DMTA Ledger",
        category="outputs",
        python_symbol="geneuclidean.sdk.credibility.build_credibility_dossier",
        execution_tier="zero_torch",
        badge="FDA/EMA 10/10",
        summary=(
            "Seals every circuit execution or multi-round DMTA loop into a chained SHA-256 "
            "DMTALedger, FDA/EMA 10-principle CredibilityDossier, and StoppingAdvisor verdict."
        ),
        inputs=(
            PortSpec(
                id="scores",
                label="Oracle Scores",
                signal_type="ORACLE_SCORES",
                direction="input",
                required=False,
                multi=True,
            ),
            PortSpec(
                id="uncertainty",
                label="Posterior UQ",
                signal_type="UNCERTAINTY_DIST",
                direction="input",
                required=False,
            ),
            PortSpec(
                id="reports",
                label="Provenance Reports",
                signal_type="PROVENANCE_REPORT",
                direction="input",
                required=False,
                multi=True,
            ),
        ),
        outputs=(
            PortSpec(
                id="dossier",
                label="Credibility Dossier",
                signal_type="PROVENANCE_REPORT",
                direction="output",
                multi=True,
                shape_hint="SHA-256 Ledger + Stopping Verdict",
            ),
        ),
        params=(
            ParamSpec(
                key="ei_saturation_tol",
                label="EI Saturation Threshold",
                type="number",
                default=0.02,
                min=0.005,
                max=0.10,
                step=0.005,
            ),
            ParamSpec(
                key="jaccard_stability_tol",
                label="Top-K Jaccard Stop",
                type="number",
                default=0.85,
                min=0.50,
                max=0.98,
                step=0.05,
            ),
            ParamSpec(
                key="export_fda_ema_dossier",
                label="Generate FDA/EMA 10/10 Dossier",
                type="boolean",
                default=True,
            ),
        ),
    ),
)

BLOCK_CATALOG: dict[str, BlockDefinition] = {b.block_type: b for b in BLOCK_DEFINITIONS}


# ---------------------------------------------------------------------------
# 4. Five Flagship One-Click End-to-End Circuit Presets
# ---------------------------------------------------------------------------

FLAGSHIP_PRESETS: tuple[dict[str, Any], ...] = (
    # Preset 1 (Default): SE(3) Pocket-Conditioned De Novo Circuit
    {
        "schemaVersion": SCHEMA_VERSION,
        "id": "se3_denovo_cns",
        "name": "SE(3) Pocket-Conditioned De Novo Circuit",
        "description": (
            "End-to-end Tourette DRD1 (9LLG) anti-EPS de novo discovery: chiral PaiNN-SE(3) -> "
            "SAFE-GPT-87M -> parallel 12-Ep Avoid-ome + AutoDock Vina v1.2.7 -> "
            "7-Axis PoseRealityAuditor + OpenFE RBFE (Δq=0) -> 3D Studio Inspector."
        ),
        "metadata": {
            "presetId": "se3_denovo_cns",
            "author": "GenEuclidean Team",
            "tags": ["se3", "de-novo", "tourette", "drd1", "openfe"],
        },
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1",
            "budget": 32,
        },
        "viewport": {"x": 20, "y": 40, "zoom": 0.78},
        "nodes": [
            {
                "id": "n_pocket",
                "blockType": "protein_pocket",
                "position": {"x": 36, "y": 170},
                "params": {
                    "target_id": "DRD1",
                    "pocket_cutoff": 10.0,
                    "max_atoms": 384,
                    "keep_cofactors": True,
                },
            },
            {
                "id": "n_painn",
                "blockType": "painn_se3",
                "position": {"x": 308, "y": 64},
                "params": {
                    "num_layers": 4,
                    "hidden_dim": 192,
                    "vector_dim": 64,
                    "out_dim": 256,
                    "num_rbf": 24,
                    "cutoff": 5.5,
                    "max_neighbors": 24,
                    "use_se3_chirality": True,
                },
            },
            {
                "id": "n_safegpt",
                "blockType": "safe_gpt",
                "position": {"x": 580, "y": 64},
                "params": {
                    "mode": "de_novo",
                    "num_samples": 32,
                    "strategy": "min_p",
                    "cfg_scale": 1.6,
                    "temperature": 0.95,
                    "min_p": 0.08,
                    "num_queries": 16,
                    "core_smiles_with_dummy": "",
                },
            },
            {
                "id": "n_admet",
                "blockType": "cns_admet_suite",
                "position": {"x": 852, "y": 64},
                "params": {
                    "property_profile": "cns_mpo_v2",
                    "admet_profile": "v4_12ep",
                    "min_mpo_v2": 4.5,
                    "max_herg_prob": 0.30,
                    "require_triage_pass": True,
                },
            },
            {
                "id": "n_vina",
                "blockType": "autodock_vina",
                "position": {"x": 852, "y": 276},
                "params": {
                    "exhaustiveness": 8,
                    "num_modes": 9,
                    "box_mode": "tight_per_axis",
                    "max_ring_confs": 4,
                    "protonate_ph74": True,
                },
            },
            {
                "id": "n_openfe",
                "blockType": "openfe_rbfe_exporter",
                "position": {"x": 1124, "y": 64},
                "params": {
                    "require_same_charge": True,
                    "filter_ring_strain": True,
                    "max_ring_strain_kcal": 3.0,
                    "add_cycle_closures": True,
                    "out_dir": "results/openfe_drd1_denovo",
                },
            },
            {
                "id": "n_audit",
                "blockType": "pose_reality_auditor",
                "position": {"x": 1124, "y": 276},
                "params": {
                    "property_profile": "cns_mpo_v2",
                    "ring_strain_threshold_kcal": 3.0,
                    "torsion_barrier_threshold_kcal": 2.5,
                    "torsion_max_red": 0,
                    "strict_input_charge": False,
                },
            },
            {
                "id": "n_studio3d",
                "blockType": "studio_3d_inspector",
                "position": {"x": 1396, "y": 170},
                "params": {
                    "theme": "dark",
                    "show_native_ghost": True,
                    "highlight_anchors": True,
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_painn",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e2",
                "source": "n_painn",
                "sourcePort": "pocket_graph",
                "target": "n_safegpt",
                "targetPort": "pocket_graph",
                "signalType": "SE3_GRAPH",
                "kind": "forward",
            },
            {
                "id": "e3",
                "source": "n_safegpt",
                "sourcePort": "molecules",
                "target": "n_admet",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e4",
                "source": "n_safegpt",
                "sourcePort": "molecules",
                "target": "n_vina",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e5",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_vina",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e6",
                "source": "n_admet",
                "sourcePort": "safe_mols",
                "target": "n_openfe",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e7",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_openfe",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e8",
                "source": "n_vina",
                "sourcePort": "poses",
                "target": "n_audit",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e9",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_audit",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e10",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_studio3d",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e11",
                "source": "n_audit",
                "sourcePort": "verified_poses",
                "target": "n_studio3d",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e12",
                "source": "n_admet",
                "sourcePort": "admet_scores",
                "target": "n_studio3d",
                "targetPort": "scores",
                "signalType": "ORACLE_SCORES",
                "kind": "forward",
            },
            {
                "id": "e13",
                "source": "n_audit",
                "sourcePort": "audit_report",
                "target": "n_studio3d",
                "targetPort": "audit_report",
                "signalType": "PROVENANCE_REPORT",
                "kind": "forward",
            },
        ],
    },
    # Preset 2: Multi-Fidelity Active Learning DMTA Closed Loop
    {
        "schemaVersion": SCHEMA_VERSION,
        "id": "al_dmta_closed_loop",
        "name": "Multi-Fidelity Active Learning DMTA Closed Loop",
        "description": (
            "Closed-loop Lab-in-a-Loop over 400 Bemis–Murcko DRD1 scaffolds using TanimotoGP -> "
            "Split-Conformal UQ -> KrigingBeliever -> MultiFidelityRouter -> AutoDock Vina with "
            "an explicit DMTA Feedback Loop updating the GP posterior each round + DMTALedger."
        ),
        "metadata": {
            "presetId": "al_dmta_closed_loop",
            "author": "GenEuclidean Team",
            "tags": ["active-learning", "dmta", "feedback-loop", "conformal", "v5.1"],
        },
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 5,
            "targetContext": "DRD1",
            "budget": 48,
        },
        "viewport": {"x": 20, "y": 60, "zoom": 0.72},
        "nodes": [
            {
                "id": "n_pool",
                "blockType": "chembl_pool",
                "position": {"x": 36, "y": 64},
                "params": {
                    "dataset": "DRD1_chembl37_400",
                    "pool_size": 400,
                    "dedup_murcko": True,
                },
            },
            {
                "id": "n_pocket",
                "blockType": "protein_pocket",
                "position": {"x": 36, "y": 280},
                "params": {
                    "target_id": "DRD1",
                    "pocket_cutoff": 10.0,
                    "max_atoms": 384,
                    "keep_cofactors": True,
                },
            },
            {
                "id": "n_gp",
                "blockType": "tanimoto_gp",
                "position": {"x": 308, "y": 64},
                "params": {
                    "kernel": "tanimoto",
                    "fp_bits": 2048,
                    "noise_var": 0.05,
                    "signal_var": 1.0,
                    "cold_start": "butina_medoids",
                },
            },
            {
                "id": "n_conf",
                "blockType": "conformal_calibrator",
                "position": {"x": 580, "y": 64},
                "params": {
                    "mode": "normalized",
                    "alpha": 0.10,
                    "scaffold_grouped": True,
                },
            },
            {
                "id": "n_acq",
                "blockType": "batch_acquisition",
                "position": {"x": 852, "y": 64},
                "params": {
                    "strategy": "kriging_believer",
                    "batch_size": 12,
                    "ucb_beta": 1.5,
                    "diversity_weight": 0.30,
                    "mask_pending_async": True,
                },
            },
            {
                "id": "n_router",
                "blockType": "multifidelity_router",
                "position": {"x": 1124, "y": 170},
                "params": {
                    "tier1_budget": 12,
                    "tier2_budget": 4,
                    "cost_ratio_t2_t1": 20.0,
                    "min_spearman_rho": 0.30,
                },
            },
            {
                "id": "n_vina",
                "blockType": "autodock_vina",
                "position": {"x": 1396, "y": 64},
                "params": {
                    "exhaustiveness": 8,
                    "num_modes": 9,
                    "box_mode": "tight_per_axis",
                    "max_ring_confs": 4,
                    "protonate_ph74": True,
                },
            },
            {
                "id": "n_ledger",
                "blockType": "dmta_ledger",
                "position": {"x": 1668, "y": 170},
                "params": {
                    "ei_saturation_tol": 0.02,
                    "jaccard_stability_tol": 0.85,
                    "export_fda_ema_dossier": True,
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_gp",
                "targetPort": "pool_mols",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e2",
                "source": "n_pool",
                "sourcePort": "molecules",
                "target": "n_acq",
                "targetPort": "pool_mols",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e3",
                "source": "n_gp",
                "sourcePort": "posterior",
                "target": "n_conf",
                "targetPort": "posterior",
                "signalType": "UNCERTAINTY_DIST",
                "kind": "forward",
            },
            {
                "id": "e4",
                "source": "n_conf",
                "sourcePort": "calibrated_uq",
                "target": "n_acq",
                "targetPort": "posterior",
                "signalType": "UNCERTAINTY_DIST",
                "kind": "forward",
            },
            {
                "id": "e5",
                "source": "n_conf",
                "sourcePort": "calibrated_uq",
                "target": "n_router",
                "targetPort": "posterior",
                "signalType": "UNCERTAINTY_DIST",
                "kind": "forward",
            },
            {
                "id": "e6",
                "source": "n_acq",
                "sourcePort": "selected_batch",
                "target": "n_router",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e7",
                "source": "n_router",
                "sourcePort": "tier1_batch",
                "target": "n_vina",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e8",
                "source": "n_pocket",
                "sourcePort": "pocket",
                "target": "n_vina",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e9",
                "source": "n_vina",
                "sourcePort": "scores",
                "target": "n_ledger",
                "targetPort": "scores",
                "signalType": "ORACLE_SCORES",
                "kind": "forward",
            },
            {
                "id": "e10",
                "source": "n_conf",
                "sourcePort": "calibrated_uq",
                "target": "n_ledger",
                "targetPort": "uncertainty",
                "signalType": "UNCERTAINTY_DIST",
                "kind": "forward",
            },
            {
                "id": "e11",
                "source": "n_router",
                "sourcePort": "routing_report",
                "target": "n_ledger",
                "targetPort": "reports",
                "signalType": "PROVENANCE_REPORT",
                "kind": "forward",
            },
            {
                "id": "e_fb_1",
                "source": "n_vina",
                "sourcePort": "scores",
                "target": "n_gp",
                "targetPort": "observations",
                "signalType": "ORACLE_SCORES",
                "kind": "feedback",
                "label": "↻ DMTA Round t+1 Update",
            },
        ],
    },
    # Preset 3: Paralog Selectivity & Avoid-ome MMP Rescue
    {
        "schemaVersion": SCHEMA_VERSION,
        "id": "paralog_selectivity_mmp",
        "name": "Paralog Selectivity & Avoid-ome MMP Rescue",
        "description": (
            "Multi-conformation DRD1 vs. off-target DRD2 contrastive CFG steering + 1-step "
            "makeable reaction chemistry + Roche-style Avoid-ome 1-cut bioisostere MMP rescue + "
            "12-endpoint Avoid-ome + charge-matched OpenFE RBFE network."
        ),
        "metadata": {
            "presetId": "paralog_selectivity_mmp",
            "author": "GenEuclidean Team",
            "tags": ["selectivity", "mmp", "synthesis", "avoidome", "openfe"],
        },
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1",
            "budget": 32,
        },
        "viewport": {"x": 20, "y": 40, "zoom": 0.82},
        "nodes": [
            {
                "id": "n_ens",
                "blockType": "multiconformer_ensemble",
                "position": {"x": 36, "y": 64},
                "params": {
                    "on_target_id": "DRD1",
                    "n_conformers": 2,
                    "off_target_id": "DRD2",
                    "weighting": "uniform",
                },
            },
            {
                "id": "n_reag",
                "blockType": "reagent_library",
                "position": {"x": 36, "y": 280},
                "params": {
                    "core_smiles": "O=C(O)c1ccc2ccccc2n1",
                    "reagent_pack": "cns_amines",
                    "max_reagents": 32,
                },
            },
            {
                "id": "n_painn",
                "blockType": "painn_se3",
                "position": {"x": 308, "y": 64},
                "params": {
                    "num_layers": 4,
                    "hidden_dim": 192,
                    "vector_dim": 64,
                    "out_dim": 256,
                    "num_rbf": 24,
                    "cutoff": 5.5,
                    "max_neighbors": 24,
                    "use_se3_chirality": True,
                },
            },
            {
                "id": "n_rxn",
                "blockType": "reaction_enumerator",
                "position": {"x": 308, "y": 280},
                "params": {
                    "mode": "enumerate_and_retro",
                    "reaction_family": "amide_coupling_acid_amine",
                    "max_products": 64,
                    "triage_filter": True,
                },
            },
            {
                "id": "n_cfg",
                "blockType": "pocket_ensemble_cfg",
                "position": {"x": 580, "y": 64},
                "params": {
                    "backbone_id": "datamol-io/safe-gpt",
                    "w_on": 1.6,
                    "w_off": 0.65,
                    "num_samples": 32,
                    "compute_pmi_3d": True,
                },
            },
            {
                "id": "n_mmp",
                "blockType": "leadopt_mmp_sar",
                "position": {"x": 580, "y": 280},
                "params": {
                    "mode": "mmp_and_cliffs",
                    "property_profile": "cns_mpo_v2",
                    "max_proposals": 20,
                    "cliff_threshold": 1.0,
                    "triage_filter": True,
                },
            },
            {
                "id": "n_admet",
                "blockType": "cns_admet_suite",
                "position": {"x": 852, "y": 64},
                "params": {
                    "property_profile": "cns_mpo_v2",
                    "admet_profile": "v4_12ep",
                    "min_mpo_v2": 4.5,
                    "max_herg_prob": 0.25,
                    "require_triage_pass": True,
                },
            },
            {
                "id": "n_openfe",
                "blockType": "openfe_rbfe_exporter",
                "position": {"x": 1124, "y": 170},
                "params": {
                    "require_same_charge": True,
                    "filter_ring_strain": True,
                    "max_ring_strain_kcal": 3.0,
                    "add_cycle_closures": True,
                    "out_dir": "results/openfe_drd1_selectivity",
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n_ens",
                "sourcePort": "on_pockets",
                "target": "n_painn",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e2",
                "source": "n_painn",
                "sourcePort": "pocket_graph",
                "target": "n_cfg",
                "targetPort": "pocket_graph",
                "signalType": "SE3_GRAPH",
                "kind": "forward",
            },
            {
                "id": "e3",
                "source": "n_ens",
                "sourcePort": "off_pocket",
                "target": "n_cfg",
                "targetPort": "off_pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e4",
                "source": "n_reag",
                "sourcePort": "reagents",
                "target": "n_rxn",
                "targetPort": "reagents",
                "signalType": "REAGENTS",
                "kind": "forward",
            },
            {
                "id": "e5",
                "source": "n_rxn",
                "sourcePort": "makeable_mols",
                "target": "n_mmp",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e6",
                "source": "n_cfg",
                "sourcePort": "molecules",
                "target": "n_admet",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e7",
                "source": "n_mmp",
                "sourcePort": "rescued_mols",
                "target": "n_admet",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e8",
                "source": "n_admet",
                "sourcePort": "safe_mols",
                "target": "n_openfe",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e9",
                "source": "n_ens",
                "sourcePort": "on_pockets",
                "target": "n_openfe",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
        ],
    },
    # Preset 4: Boltz-2 / AF3 7-Axis Audit & Conformal DMPK
    {
        "schemaVersion": SCHEMA_VERSION,
        "id": "boltz2_audit_conformal_dmpk",
        "name": "Boltz-2 / AF3 7-Axis Audit & Conformal DMPK",
        "description": (
            "Zero-PyTorch industrial quality-assurance circuit auditing a Boltz-2 / AlphaFold 3 "
            "co-folded ACHE complex (4EY7) across Cremer–Pople ring strain, relaxed torsion "
            "barriers, 7-Axis PoseRealityAuditor, 90% Split-Conformal DMPK, 3D Studio, and MCP."
        ),
        "metadata": {
            "presetId": "boltz2_audit_conformal_dmpk",
            "author": "GenEuclidean Team",
            "tags": ["boltz2", "beyond-posebusters", "conformal-dmpk", "mcp", "zero-torch"],
        },
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "ACHE",
            "budget": 16,
        },
        "viewport": {"x": 20, "y": 40, "zoom": 0.88},
        "nodes": [
            {
                "id": "n_boltz",
                "blockType": "boltz2_cofold",
                "position": {"x": 36, "y": 170},
                "params": {
                    "structure_source": "ACHE_donepezil_cofold.cif",
                    "ligand_resname": "LIG",
                    "reference_smiles": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
                    "pocket_cutoff": 10.0,
                },
            },
            {
                "id": "n_strain",
                "blockType": "ring_torsion_strain",
                "position": {"x": 320, "y": 64},
                "params": {
                    "ring_strain_threshold_kcal": 3.0,
                    "ring_n_ref_confs": 24,
                    "torsion_barrier_threshold_kcal": 2.5,
                    "torsion_max_red": 0,
                    "torsion_backend": "smarts_mmff94s",
                },
            },
            {
                "id": "n_prot",
                "blockType": "protonation_ph74",
                "position": {"x": 320, "y": 280},
                "params": {
                    "audit_tautomers": True,
                    "max_tautomers": 16,
                    "tautomer_gap_threshold": 2.0,
                    "strict_input_charge": False,
                },
            },
            {
                "id": "n_audit",
                "blockType": "pose_reality_auditor",
                "position": {"x": 604, "y": 64},
                "params": {
                    "property_profile": "cns_mpo_v2",
                    "ring_strain_threshold_kcal": 3.0,
                    "torsion_barrier_threshold_kcal": 2.5,
                    "torsion_max_red": 0,
                    "strict_input_charge": False,
                },
            },
            {
                "id": "n_dmpk",
                "blockType": "conformal_dmpk",
                "position": {"x": 604, "y": 280},
                "params": {
                    "endpoint": "OpenADMET_LogD",
                    "alpha": 0.10,
                    "n_estimators": 80,
                },
            },
            {
                "id": "n_studio3d",
                "blockType": "studio_3d_inspector",
                "position": {"x": 888, "y": 64},
                "params": {
                    "theme": "dark",
                    "show_native_ghost": True,
                    "highlight_anchors": True,
                },
            },
            {
                "id": "n_mcp",
                "blockType": "mcp_tool_export",
                "position": {"x": 888, "y": 280},
                "params": {
                    "tool_name": "audit_complex",
                    "transport": "python_direct",
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n_boltz",
                "sourcePort": "pose",
                "target": "n_strain",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e2",
                "source": "n_strain",
                "sourcePort": "clean_poses",
                "target": "n_audit",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e3",
                "source": "n_boltz",
                "sourcePort": "pocket",
                "target": "n_audit",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e4",
                "source": "n_boltz",
                "sourcePort": "ligand_mols",
                "target": "n_prot",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e5",
                "source": "n_prot",
                "sourcePort": "protonated_mols",
                "target": "n_dmpk",
                "targetPort": "molecules",
                "signalType": "MOL_BATCH",
                "kind": "forward",
            },
            {
                "id": "e6",
                "source": "n_boltz",
                "sourcePort": "pocket",
                "target": "n_studio3d",
                "targetPort": "pocket",
                "signalType": "POCKET_3D",
                "kind": "forward",
            },
            {
                "id": "e7",
                "source": "n_audit",
                "sourcePort": "verified_poses",
                "target": "n_studio3d",
                "targetPort": "poses",
                "signalType": "POSE_3D",
                "kind": "forward",
            },
            {
                "id": "e8",
                "source": "n_audit",
                "sourcePort": "audit_report",
                "target": "n_studio3d",
                "targetPort": "audit_report",
                "signalType": "PROVENANCE_REPORT",
                "kind": "forward",
            },
            {
                "id": "e9",
                "source": "n_audit",
                "sourcePort": "audit_report",
                "target": "n_mcp",
                "targetPort": "audit_report",
                "signalType": "PROVENANCE_REPORT",
                "kind": "forward",
            },
            {
                "id": "e10",
                "source": "n_dmpk",
                "sourcePort": "dmpk_intervals",
                "target": "n_mcp",
                "targetPort": "uncertainty",
                "signalType": "UNCERTAINTY_DIST",
                "kind": "forward",
            },
        ],
    },
    # Preset 5: Blank Breadboard
    {
        "schemaVersion": SCHEMA_VERSION,
        "id": "blank_breadboard",
        "name": "Blank Breadboard",
        "description": (
            "Empty GenEuclidean breadboard ready for custom drug-design circuits — drag any "
            "block from the dock or wire a custom Combinatorial Thompson Sampling loop."
        ),
        "metadata": {
            "presetId": "blank_breadboard",
            "author": "GenEuclidean Team",
            "tags": ["blank", "custom"],
        },
        "execution": {
            "mode": "local_python",
            "seed": 42,
            "dmtaRounds": 1,
            "targetContext": "DRD1",
            "budget": 16,
        },
        "viewport": {"x": 0, "y": 0, "zoom": 1.0},
        "nodes": [],
        "edges": [],
    },
)

PRESET_BY_ID: dict[str, dict[str, Any]] = {p["id"]: p for p in FLAGSHIP_PRESETS}


def get_block_definition(block_type: str) -> BlockDefinition:
    """Look up an authoritative :class:`BlockDefinition` by ``blockType``."""
    try:
        return BLOCK_CATALOG[block_type]
    except KeyError as exc:
        raise KeyError(
            f"Unknown circuit blockType {block_type!r}. "
            f"Available ({len(BLOCK_CATALOG)}): {sorted(BLOCK_CATALOG)}"
        ) from exc


def list_block_definitions(category: str | None = None) -> tuple[BlockDefinition, ...]:
    """Return all 36 block definitions, optionally filtered by ``category``."""
    if category is None:
        return BLOCK_DEFINITIONS
    return tuple(b for b in BLOCK_DEFINITIONS if b.category == category)


def get_preset(preset_id: str) -> dict[str, Any]:
    """Return a fresh deep copy of one of the 5 flagship ``CircuitGraphV1`` presets."""
    if preset_id not in PRESET_BY_ID:
        raise KeyError(
            f"Unknown preset {preset_id!r}. Available presets: {list(PRESET_BY_ID.keys())}"
        )
    return copy.deepcopy(PRESET_BY_ID[preset_id])


def list_presets() -> list[dict[str, Any]]:
    """Return fresh deep copies of all 5 flagship ``CircuitGraphV1`` presets."""
    return [copy.deepcopy(p) for p in FLAGSHIP_PRESETS]


def export_catalog_payload() -> dict[str, Any]:
    """Build the complete JSON payload returned by ``GET /api/catalog``."""
    return {
        "schemaVersion": SCHEMA_VERSION,
        "signalTypes": [s.to_dict() for s in SIGNAL_SPECS],
        "categories": [c.to_dict() for c in CATEGORY_SPECS],
        "blocks": [b.to_dict() for b in BLOCK_DEFINITIONS],
        "presets": list_presets(),
    }


# Canonical JSON-dict views & aliases for zero-overhead validator / server lookups
SIGNAL_TYPE_IDS: frozenset[str] = VALID_SIGNAL_TYPES
BLOCK_MAP: dict[str, dict[str, Any]] = {b.block_type: b.to_dict() for b in BLOCK_DEFINITIONS}
get_wiring_remediation_hint = get_mismatch_remediation
get_catalog_payload = export_catalog_payload
get_studio_catalog = export_catalog_payload


__all__ = [
    "BLOCK_CATALOG",
    "BLOCK_DEFINITIONS",
    "BLOCK_MAP",
    "CATEGORY_SPECS",
    "FLAGSHIP_PRESETS",
    "PRESET_BY_ID",
    "REMEDIATION_HINTS",
    "SCHEMA_VERSION",
    "SIGNAL_SPECS",
    "SIGNAL_TYPE_IDS",
    "VALID_SIGNAL_TYPES",
    "BlockCategoryId",
    "BlockDefinition",
    "CategorySpec",
    "EdgeKind",
    "ExecutionTier",
    "ParamSpec",
    "ParamType",
    "PortDirection",
    "PortSpec",
    "SignalSpec",
    "SignalType",
    "export_catalog_payload",
    "get_block_definition",
    "get_catalog_payload",
    "get_mismatch_remediation",
    "get_preset",
    "get_studio_catalog",
    "get_wiring_remediation_hint",
    "list_block_definitions",
    "list_presets",
]
