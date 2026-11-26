import type {
  BlockCategoryId,
  BlockCategoryMeta,
  BlockDefinition,
  CircuitGraphV1,
  CircuitNode,
  CircuitPreset,
  SignalType,
  SignalTypeMeta,
} from '../types/circuit';

export const SCHEMA_VERSION = 'geneuclidean.circuit.v1' as const;

export const SIGNAL_TYPE_META: Record<SignalType, SignalTypeMeta> = {
  POCKET_3D: {
    type: 'POCKET_3D',
    label: '3D Pocket',
    shortLabel: 'Pocket',
    darkHex: '#38BDF8',
    lightHex: '#0284C7',
    pythonContract: 'ge.Pocket | Sequence[ge.Pocket] ([N, 3], [N, 52])',
  },
  SE3_GRAPH: {
    type: 'SE3_GRAPH',
    label: 'SE(3) Graph',
    shortLabel: 'SE(3)',
    darkHex: '#818CF8',
    lightHex: '#4F46E5',
    pythonContract: 'PocketEncoderOutput (node [B,N,256], vec [B,N,64,3])',
  },
  LATENT_VEC: {
    type: 'LATENT_VEC',
    label: 'Latent Vector',
    shortLabel: 'Latent',
    darkHex: '#A78BFA',
    lightHex: '#7C3AED',
    pythonContract: 'NDArray[f16/f32] | Tensor ([N, 128] / [N, 768] / [B,24,d])',
  },
  MOL_BATCH: {
    type: 'MOL_BATCH',
    label: 'Molecules',
    shortLabel: 'Mols',
    darkHex: '#34D399',
    lightHex: '#059669',
    pythonContract: 'list[str] canonical SMILES / DesignResult / ScreeningHit[]',
  },
  POSE_3D: {
    type: 'POSE_3D',
    label: '3D Pose',
    shortLabel: 'Pose 3D',
    darkHex: '#2DD4BF',
    lightHex: '#0D9488',
    pythonContract: 'list[DockingPoseResult] | list[Chem.Mol] (3D SDF + PDB)',
  },
  ORACLE_SCORES: {
    type: 'ORACLE_SCORES',
    label: 'Oracle Scores',
    shortLabel: 'Scores',
    darkHex: '#FBBF24',
    lightHex: '#D97706',
    pythonContract: 'dict[str, dict[str, float]] (Vina ΔG, pKd, MPO, Avoid-ome)',
  },
  UNCERTAINTY_DIST: {
    type: 'UNCERTAINTY_DIST',
    label: 'Uncertainty Dist',
    shortLabel: 'UQ Dist',
    darkHex: '#F472B6',
    lightHex: '#DB2777',
    pythonContract: 'SurrogatePrediction (μ, σ_epi, σ_ale, Conformal [lo, hi])',
  },
  REAGENTS: {
    type: 'REAGENTS',
    label: 'Reagents',
    shortLabel: 'Reagents',
    darkHex: '#FB923C',
    lightHex: '#EA580C',
    pythonContract: 'dict(core_smiles, building_blocks, reactions[16])',
  },
  RBFE_GRAPH: {
    type: 'RBFE_GRAPH',
    label: 'RBFE Network',
    shortLabel: 'RBFE',
    darkHex: '#FB7185',
    lightHex: '#E11D48',
    pythonContract: 'OpenFENetworkBundle (Lomap graph, Δq = 0, aligned SDF)',
  },
  PROVENANCE_REPORT: {
    type: 'PROVENANCE_REPORT',
    label: 'Provenance',
    shortLabel: 'Report',
    darkHex: '#CBD5E1',
    lightHex: '#475569',
    pythonContract: 'PoseAuditReport | CredibilityDossier | SHA-256 DMTALedger',
  },
};

export const BLOCK_CATEGORIES: BlockCategoryMeta[] = [
  {
    id: 'inputs',
    label: 'Inputs & Structures',
    shortLabel: 'Inputs',
    darkHex: '#38BDF8',
    lightHex: '#0284C7',
    description: '28 curated 3D CNS pockets, multi-conformer ensembles, reagents, Boltz-2/AF3 & ChEMBL pools.',
  },
  {
    id: 'encoders',
    label: 'SE(3) / 3D Neural Encoders',
    shortLabel: 'Encoders',
    darkHex: '#818CF8',
    lightHex: '#4F46E5',
    description: 'Chiral PaiNN-SE(3), EGNN-3D, E3NN Irreps, DrugCLIP hypersphere, and IBM MMELON (84M).',
  },
  {
    id: 'generators',
    label: 'De Novo Generators & Steering',
    shortLabel: 'Generators',
    darkHex: '#34D399',
    lightHex: '#059669',
    description: 'SAFE-GPT-87M, GP-MoLFormer / NovoMolGen, contrastive multi-pocket CFG, and MTL alignment.',
  },
  {
    id: 'screening',
    label: 'Virtual Screening & Retrieval',
    shortLabel: 'Screening',
    darkHex: '#A78BFA',
    lightHex: '#7C3AED',
    description: 'Float16 LigandVectorIndex, 0.02 ms contrastive PocketScreener, pKd rerank & catalog grounding.',
  },
  {
    id: 'active_learning',
    label: 'Active Learning DMTA',
    shortLabel: 'Active Learning',
    darkHex: '#F472B6',
    lightHex: '#DB2777',
    description: 'TanimotoGP, Ensemble, Split-Conformal UQ, 12 batch acquisitions, Multi-Fidelity & Combinatorial TS.',
  },
  {
    id: 'physics_audit',
    label: '3D Physics & 7-Axis Pose Reality',
    shortLabel: '3D Physics',
    darkHex: '#2DD4BF',
    lightHex: '#0D9488',
    description: 'pH 7.4 ionization, AutoDock Vina v1.2.7, Cremer–Pople ring/torsion strain & 7-Axis PoseRealityAuditor.',
  },
  {
    id: 'leadopt_admet',
    label: 'Lead-Opt, ADMET & Synthesis',
    shortLabel: 'Lead-Opt',
    darkHex: '#FBBF24',
    lightHex: '#D97706',
    description: '16-reaction SMIRKS makeability, Avoid-ome MMP + R-group cliffs, 12-Ep Avoid-ome & Conformal DMPK.',
  },
  {
    id: 'outputs',
    label: 'Outputs & Open-Pharma Export',
    shortLabel: 'Outputs',
    darkHex: '#FB7185',
    lightHex: '#E11D48',
    description: 'Charge-matched OpenFE RBFE networks, 3D Studio WebGL, typed MCP tools & SHA-256 DMTA Ledger.',
  },
];

export const CATEGORY_META_MAP: Record<BlockCategoryId, BlockCategoryMeta> = Object.fromEntries(
  BLOCK_CATEGORIES.map((c) => [c.id, c]),
) as Record<BlockCategoryId, BlockCategoryMeta>;

const ALL_28_TARGETS = [
  'DRD1',
  'DRD2',
  'DRD3',
  'VMAT2',
  'ADRA2A',
  'HRH3',
  'PDE10A',
  'CHRM4',
  'HTR2A',
  'SERT',
  'TAAR1',
  'DAT',
  'ACHE',
  'MAOB',
  'NLRP3',
  'OGA',
  'GSK3B',
  'BACE1',
  'CHRM1',
  'HTR1A',
  'GLYT1',
  'GPR52',
  'MGLL',
  'PDE1B',
  'GABRA5',
  'GAT1',
  'ADORA2A',
  'GRM5',
];

export const TARGET_PDB_MAP: Record<string, string> = {
  DRD1: '9LLG',
  DRD2: '6CM4',
  DRD3: '3PBL',
  VMAT2: '8JTC',
  ADRA2A: '6KUX',
  HRH3: '7F61',
  PDE10A: '4AEL',
  CHRM4: '8FX5',
  HTR2A: '6WHA',
  SERT: '6AWO',
  TAAR1: '8W89',
  DAT: '8Y2G',
  ACHE: '4EY7',
  MAOB: '2V5Z',
  NLRP3: '7ALV',
  OGA: '5M7S',
  GSK3B: '1Q5K',
  BACE1: '2ZJM',
  CHRM1: '5CXV',
  HTR1A: '7E2Z',
  GLYT1: '6ZBV',
  GPR52: '6LI3',
  MGLL: '3PE6',
  PDE1B: '5B25',
  GABRA5: '8BHK',
  GAT1: '7Y7W',
  ADORA2A: '4EIY',
  GRM5: '4OO9',
};

export const BLOCK_CATALOG: BlockDefinition[] = [
  // =========================================================================
  // Category 1 — Inputs & Structures (5 blocks)
  // =========================================================================
  {
    blockType: 'protein_pocket',
    name: 'Protein Pocket',
    shortName: 'Protein Pocket',
    category: 'inputs',
    pythonSymbol: 'geneuclidean.sdk.pocket.Pocket.from_target',
    executionTier: 'zero_torch',
    badge: '28 Targets',
    summary:
      'Loads any of the 28 curated human CNS/Neuro 3D pockets (preserving catalytic cofactors FAD/ADP/Zn/Mg) or a local PDB/mmCIF structure.',
    inputs: [],
    outputs: [
      {
        id: 'pocket',
        label: 'pocket',
        signalType: 'POCKET_3D',
        direction: 'output',
        multi: true,
        shapeHint: 'Pocket[N×3, N×52]',
      },
    ],
    params: [
      {
        key: 'target_id',
        label: 'Human 3D Target',
        type: 'select',
        default: 'DRD1',
        options: ALL_28_TARGETS,
        description: 'Curated human target from the 28-target Tourette/SCZ/Alzheimer atlas.',
      },
      {
        key: 'pocket_cutoff',
        label: 'Pocket Radius',
        type: 'number',
        default: 10.0,
        min: 6.0,
        max: 16.0,
        step: 0.5,
        unit: 'Å',
      },
      {
        key: 'max_atoms',
        label: 'Max Heavy Atoms',
        type: 'number',
        default: 384,
        min: 128,
        max: 512,
        step: 32,
      },
      {
        key: 'keep_cofactors',
        label: 'Preserve Cofactors (FAD/ADP/Zn/Mg)',
        type: 'boolean',
        default: true,
      },
    ],
  },
  {
    blockType: 'multiconformer_ensemble',
    name: 'Multi-Conformer Ensemble',
    shortName: 'Conformer Ensemble',
    category: 'inputs',
    pythonSymbol: 'geneuclidean.sdk.pocket.Pocket.from_target',
    executionTier: 'zero_torch',
    badge: 'K-Holo',
    summary:
      'Bundles K experimental holo/apo conformations of an on-target pocket plus an optional off-target paralog counter-pocket for contrastive selectivity.',
    inputs: [],
    outputs: [
      {
        id: 'on_pockets',
        label: 'on_pockets',
        signalType: 'POCKET_3D',
        direction: 'output',
        multi: true,
        shapeHint: 'Ensemble[K×Pocket]',
      },
      {
        id: 'off_pocket',
        label: 'off_pocket',
        signalType: 'POCKET_3D',
        direction: 'output',
        multi: true,
        shapeHint: 'CounterPocket',
      },
    ],
    params: [
      {
        key: 'on_target_id',
        label: 'On-Target Pocket',
        type: 'select',
        default: 'DRD1',
        options: ['DRD1', 'ACHE', 'MAOB', 'BACE1', 'CHRM4', 'SERT', 'GRM5', 'GABRA5'],
      },
      {
        key: 'n_conformers',
        label: 'Conformations (K)',
        type: 'number',
        default: 2,
        min: 1,
        max: 4,
        step: 1,
      },
      {
        key: 'off_target_id',
        label: 'Off-Target Counter-Pocket',
        type: 'select',
        default: 'DRD2',
        options: ['DRD2', 'BCHE', 'MAOA', 'BACE2', 'CHRM3', 'DAT', 'NONE'],
      },
      {
        key: 'weighting',
        label: 'Ensemble Weighting',
        type: 'select',
        default: 'uniform',
        options: ['uniform', 'resolution_weighted'],
      },
    ],
  },
  {
    blockType: 'reagent_library',
    name: 'Reagent Library',
    shortName: 'Reagent Library',
    category: 'inputs',
    pythonSymbol: 'geneuclidean.synthesis.ROBUST_MEDCHEM_REACTIONS',
    executionTier: 'zero_torch',
    badge: '16 SMIRKS',
    summary:
      'Supplies a medicinal-chemistry core scaffold and curated Enamine-style coupling building blocks for makeable enumeration and Combinatorial TS.',
    inputs: [],
    outputs: [
      {
        id: 'reagents',
        label: 'reagents',
        signalType: 'REAGENTS',
        direction: 'output',
        multi: true,
        shapeHint: 'Core + N_B Blocks',
      },
      {
        id: 'core_mols',
        label: 'core_mols',
        signalType: 'MOL_BATCH',
        direction: 'output',
        multi: true,
        shapeHint: '1 Core SMILES',
      },
    ],
    params: [
      {
        key: 'core_smiles',
        label: 'Core Scaffold SMILES',
        type: 'string',
        default: 'O=C(O)c1ccc2ccccc2n1',
      },
      {
        key: 'reagent_pack',
        label: 'Building-Block Pack',
        type: 'select',
        default: 'cns_amines',
        options: ['cns_amines', 'aryl_boronics', 'sulfonyl_chlorides', 'heteroaryl_halides'],
      },
      {
        key: 'max_reagents',
        label: 'Max Building Blocks',
        type: 'number',
        default: 32,
        min: 4,
        max: 256,
        step: 4,
      },
    ],
  },
  {
    blockType: 'boltz2_cofold',
    name: 'Boltz-2 / AF3 Co-Fold',
    shortName: 'Boltz-2 Co-Fold',
    category: 'inputs',
    pythonSymbol: 'geneuclidean.interop.boltz_adapter.load_cofolded_complex',
    executionTier: 'zero_torch',
    badge: 'mmCIF',
    summary:
      'Parses Boltz-2, AlphaFold 3, or Chai-1 .cif/.pdb co-folded predictions, restores exact bond orders from SMILES templates, and emits aligned 3D signals.',
    inputs: [],
    outputs: [
      {
        id: 'pocket',
        label: 'pocket',
        signalType: 'POCKET_3D',
        direction: 'output',
        multi: true,
        shapeHint: 'Aligned Pocket',
      },
      {
        id: 'pose',
        label: 'pose',
        signalType: 'POSE_3D',
        direction: 'output',
        multi: true,
        shapeHint: 'CoFolded Complex',
      },
      {
        id: 'ligand_mols',
        label: 'ligand_mols',
        signalType: 'MOL_BATCH',
        direction: 'output',
        multi: true,
        shapeHint: 'Extracted SMILES',
      },
    ],
    params: [
      {
        key: 'structure_source',
        label: 'Co-Folded Structure',
        type: 'select',
        default: 'ACHE_donepezil_cofold.cif',
        options: [
          'ACHE_donepezil_cofold.cif',
          'DRD1_ecopipam_cofold.cif',
          'MAOB_safinamide_cofold.cif',
          'CHRM4_xanomeline_cofold.cif',
          'custom_path',
        ],
      },
      {
        key: 'ligand_resname',
        label: 'Ligand Resname',
        type: 'string',
        default: 'LIG',
      },
      {
        key: 'reference_smiles',
        label: 'Template SMILES',
        type: 'string',
        default: 'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2',
      },
      {
        key: 'pocket_cutoff',
        label: 'Pocket Radius',
        type: 'number',
        default: 10.0,
        min: 6.0,
        max: 15.0,
        step: 0.5,
        unit: 'Å',
      },
    ],
  },
  {
    blockType: 'chembl_pool',
    name: 'ChEMBL / ZINC Pool',
    shortName: 'ChEMBL Pool',
    category: 'inputs',
    pythonSymbol: 'geneuclidean.data.splits.compute_murcko_scaffold',
    executionTier: 'zero_torch',
    badge: 'Murcko',
    summary:
      'Streams a Bemis–Murcko scaffold-deduplicated human ChEMBL 37 or ZINC-250k compound deck for zero-leakage screening and Active Learning DMTA.',
    inputs: [],
    outputs: [
      {
        id: 'molecules',
        label: 'molecules',
        signalType: 'MOL_BATCH',
        direction: 'output',
        multi: true,
        shapeHint: 'Pool[N scaffolds]',
      },
    ],
    params: [
      {
        key: 'dataset',
        label: 'Bioactivity Deck',
        type: 'select',
        default: 'DRD1_chembl37_400',
        options: [
          'DRD1_chembl37_400',
          'ACHE_chembl37_300',
          'MAOB_chembl37_300',
          'BACE1_chembl37_300',
          'SERT_chembl37_300',
          'ZINC250k_diverse',
        ],
      },
      {
        key: 'pool_size',
        label: 'Pool Size (N)',
        type: 'number',
        default: 400,
        min: 32,
        max: 5000,
        step: 32,
      },
      {
        key: 'dedup_murcko',
        label: 'Strict Bemis–Murcko Dedup',
        type: 'boolean',
        default: true,
      },
    ],
  },

  // =========================================================================
  // Category 2 — SE(3) / 3D Neural Encoders (5 blocks)
  // =========================================================================
  {
    blockType: 'painn_se3',
    name: 'PaiNN-SE(3)',
    shortName: 'PaiNN-SE(3)',
    category: 'encoders',
    pythonSymbol: 'geneuclidean.models.encoders.PaiNNEncoder',
    executionTier: 'optional_torch',
    badge: 'SE(3) · 4L',
    summary:
      'Chiral float32 SE(3)-equivariant Polarizable Atom Interaction Network (SO(3) error < 8e-6, frozen backbone retaining CASF-2016 pKd r = 0.746).',
    inputs: [
      {
        id: 'pocket',
        label: 'pocket',
        signalType: 'POCKET_3D',
        direction: 'input',
        required: true,
        multi: false,
        shapeHint: 'Pocket[N×3, N×52]',
      },
    ],
    outputs: [
      {
        id: 'pocket_graph',
        label: 'pocket_graph',
        signalType: 'SE3_GRAPH',
        direction: 'output',
        multi: true,
        shapeHint: '[B, N, 256] + [B, N, 64, 3]',
      },
      {
        id: 'pocket_vec',
        label: 'pocket_vec',
        signalType: 'LATENT_VEC',
        direction: 'output',
        multi: true,
        shapeHint: '[B, 256]',
      },
    ],
    params: [
      { key: 'num_layers', label: 'Message Layers', type: 'number', default: 4, min: 2, max: 8, step: 1 },
      { key: 'hidden_dim', label: 'Scalar Dim', type: 'select', default: 192, options: [128, 192, 256, 384] },
      { key: 'vector_dim', label: 'Vector Dim (l=1)', type: 'select', default: 64, options: [32, 64, 128] },
      { key: 'out_dim', label: 'Output Dim', type: 'select', default: 256, options: [128, 256, 512] },
      { key: 'num_rbf', label: 'Bessel RBFs', type: 'number', default: 24, min: 8, max: 64, step: 8 },
      { key: 'cutoff', label: 'Radius Cutoff', type: 'number', default: 5.5, min: 3.5, max: 8.0, step: 0.5, unit: 'Å' },
      { key: 'max_neighbors', label: 'Max kNN', type: 'number', default: 24, min: 8, max: 48, step: 4 },
      { key: 'use_se3_chirality', label: 'SE(3) Chiral Cross-Product', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'egnn_3d',
    name: 'EGNN-3D',
    shortName: 'EGNN-3D',
    category: 'encoders',
    pythonSymbol: 'geneuclidean.models.encoders.EGNNEncoder',
    executionTier: 'optional_torch',
    badge: 'E(n) · 4L',
    summary:
      'E(n)-equivariant graph neural network with radial Bessel edge updates and coordinate-equivariant message passing.',
    inputs: [
      {
        id: 'pocket',
        label: 'pocket',
        signalType: 'POCKET_3D',
        direction: 'input',
        required: true,
        multi: false,
      },
    ],
    outputs: [
      {
        id: 'pocket_graph',
        label: 'pocket_graph',
        signalType: 'SE3_GRAPH',
        direction: 'output',
        multi: true,
        shapeHint: '[B, N, 256]',
      },
      {
        id: 'pocket_vec',
        label: 'pocket_vec',
        signalType: 'LATENT_VEC',
        direction: 'output',
        multi: true,
        shapeHint: '[B, 256]',
      },
    ],
    params: [
      { key: 'num_layers', label: 'Layers', type: 'number', default: 4, min: 2, max: 8, step: 1 },
      { key: 'hidden_dim', label: 'Hidden Dim', type: 'select', default: 192, options: [128, 192, 256] },
      { key: 'out_dim', label: 'Output Dim', type: 'select', default: 256, options: [128, 256, 512] },
      { key: 'cutoff', label: 'Cutoff', type: 'number', default: 5.5, min: 3.5, max: 8.0, step: 0.5, unit: 'Å' },
      { key: 'max_neighbors', label: 'Max kNN', type: 'number', default: 24, min: 8, max: 48, step: 4 },
    ],
  },
  {
    blockType: 'e3nn_irreps',
    name: 'E3NN Irreps',
    shortName: 'E3NN Irreps',
    category: 'encoders',
    pythonSymbol: 'geneuclidean.models.encoders.E3NNEncoder',
    executionTier: 'optional_torch',
    badge: 'O(3) Tensor',
    summary:
      'Full spherical-harmonics irreducible-representation (o3.Irreps) tensor-product 3D pocket encoder with IrrepsEquivariantRMSNorm.',
    inputs: [
      {
        id: 'pocket',
        label: 'pocket',
        signalType: 'POCKET_3D',
        direction: 'input',
        required: true,
        multi: false,
      },
    ],
    outputs: [
      {
        id: 'pocket_graph',
        label: 'pocket_graph',
        signalType: 'SE3_GRAPH',
        direction: 'output',
        multi: true,
        shapeHint: 'Irreps[l≤lmax] → 256D',
      },
      {
        id: 'pocket_vec',
        label: 'pocket_vec',
        signalType: 'LATENT_VEC',
        direction: 'output',
        multi: true,
        shapeHint: '[B, 256]',
      },
    ],
    params: [
      { key: 'num_layers', label: 'Tensor Layers', type: 'number', default: 4, min: 2, max: 6, step: 1 },
      { key: 'hidden_dim', label: 'Hidden Dim', type: 'select', default: 192, options: [128, 192, 256] },
      { key: 'e3nn_lmax', label: 'Spherical Harmonic l_max', type: 'select', default: 1, options: [1, 2] },
      { key: 'use_se3_chirality', label: 'SE(3) Chiral Parity', type: 'boolean', default: true },
      { key: 'cutoff', label: 'Cutoff', type: 'number', default: 5.5, min: 3.5, max: 8.0, step: 0.5, unit: 'Å' },
    ],
  },
  {
    blockType: 'drugclip_encoder',
    name: 'DrugCLIP Encoder',
    shortName: 'DrugCLIP',
    category: 'encoders',
    pythonSymbol: 'geneuclidean.screening.encoders.DrugCLIPLigandEncoder',
    executionTier: 'optional_torch',
    badge: 'R@5 62%',
    summary:
      'Projects ligands into GenEuclidean’s 3D-pocket-aligned unit hypersphere (CASF R@5 = 62.1%, ROC-AUC = 0.937) or pure-RDKit ECFP4 baseline.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      {
        id: 'embeddings',
        label: 'embeddings',
        signalType: 'LATENT_VEC',
        direction: 'output',
        multi: true,
        shapeHint: '[N, 128] float16',
      },
    ],
    params: [
      { key: 'backend', label: 'Embedding Backend', type: 'select', default: 'drugclip_v2', options: ['drugclip_v2', 'ecfp4_binary'] },
      { key: 'clip_dim', label: 'CLIP Dim', type: 'select', default: 128, options: [128, 256, 1024] },
      { key: 'batch_size', label: 'Batch Size', type: 'number', default: 2048, min: 64, max: 8192, step: 64 },
    ],
  },
  {
    blockType: 'ibm_mmelon',
    name: 'IBM MMELON (84M)',
    shortName: 'IBM MMELON',
    category: 'encoders',
    pythonSymbol: 'geneuclidean.screening.encoders.IBMMMelonLigandEncoder',
    executionTier: 'optional_torch',
    badge: '84M · 768D',
    summary:
      'Apache-2.0 IBM Research multi-view molecular foundation encoder (84M params, 768-D unit embeddings) for hybrid 2D/SMILES + 3D screening.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      {
        id: 'embeddings',
        label: 'embeddings',
        signalType: 'LATENT_VEC',
        direction: 'output',
        multi: true,
        shapeHint: '[N, 768] float32',
      },
    ],
    params: [
      { key: 'model_id', label: 'HF Model ID', type: 'string', default: 'ibm-research/biomed.sm.mv-te-84m' },
      { key: 'batch_size', label: 'Batch Size', type: 'number', default: 64, min: 16, max: 256, step: 16 },
      { key: 'device', label: 'Device', type: 'select', default: 'auto', options: ['auto', 'mps', 'cuda', 'cpu'] },
    ],
  },

  // =========================================================================
  // Category 3 — De Novo Generators & Steering (4 blocks)
  // =========================================================================
  {
    blockType: 'safe_gpt',
    name: 'SAFE-GPT (87M)',
    shortName: 'SAFE-GPT 87M',
    category: 'generators',
    pythonSymbol: 'geneuclidean.models.llm_bridge.PocketConditionedLLM',
    executionTier: 'optional_torch',
    badge: '5.5× Safe',
    summary:
      'Fragment-modular SAFE chemical transformer (87.3M, 24-token Perceiver bridge, 5.5× v4 Pharma-Safe, 0% DCU) for de novo & scaffold decoration.',
    inputs: [
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: true, multi: false },
      { id: 'seed_mols', label: 'seed_mols', signalType: 'MOL_BATCH', direction: 'input', required: false, multi: false },
      {
        id: 'feedback_scores',
        label: 'feedback',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Batch[n SMILES]' },
      { id: 'latent', label: 'latent', signalType: 'LATENT_VEC', direction: 'output', multi: true, shapeHint: '[B, 24, 768]' },
    ],
    params: [
      { key: 'mode', label: 'Generation Mode', type: 'select', default: 'de_novo', options: ['de_novo', 'decorate_scaffold'] },
      { key: 'num_samples', label: 'Batch Samples (n)', type: 'number', default: 32, min: 4, max: 256, step: 4 },
      { key: 'strategy', label: 'Sampling Strategy', type: 'select', default: 'min_p', options: ['min_p', 'top_p', 'top_k', 'typical', 'temperature'] },
      { key: 'cfg_scale', label: 'CFG Guidance w', type: 'number', default: 1.6, min: 0.5, max: 3.0, step: 0.05 },
      { key: 'temperature', label: 'Temperature (T)', type: 'number', default: 0.95, min: 0.5, max: 1.4, step: 0.05 },
      { key: 'min_p', label: 'Min-p Threshold', type: 'number', default: 0.08, min: 0.01, max: 0.3, step: 0.01 },
      { key: 'num_queries', label: 'Perceiver Queries Q', type: 'number', default: 16, min: 4, max: 32, step: 4 },
      { key: 'core_smiles_with_dummy', label: 'Core Scaffold ([*] for Decorate)', type: 'string', default: '' },
    ],
  },
  {
    blockType: 'chemical_llm',
    name: 'Chemical Foundation LLM',
    shortName: 'Chemical LLM',
    category: 'generators',
    pythonSymbol: 'geneuclidean.models.hf_backbones.load_chemical_backbone',
    executionTier: 'optional_torch',
    badge: '47–300M',
    summary:
      'Pluggable 3D-pocket-conditioned SMILES generator supporting GP-MoLFormer-Uniq, NovoMolGen-157M/300M, gpt2_zinc_87m, and PocketTransformer.',
    inputs: [
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: true, multi: false },
      {
        id: 'feedback_scores',
        label: 'feedback',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Batch[n SMILES]' },
      { id: 'latent', label: 'latent', signalType: 'LATENT_VEC', direction: 'output', multi: true, shapeHint: '[B, 24, d_llm]' },
    ],
    params: [
      {
        key: 'backbone_id',
        label: 'Chemical LLM Backbone',
        type: 'select',
        default: 'ibm-research/GP-MoLFormer-Uniq',
        options: [
          'ibm-research/GP-MoLFormer-Uniq',
          'chandar-lab/NovoMolGen_157M_SMILES_BPE',
          'chandar-lab/NovoMolGen_300M_SMILES_BPE',
          'entropy/gpt2_zinc_87m',
          'v2-PocketTransformerDecoder',
          'mock-tiny-gpt2',
        ],
      },
      { key: 'num_samples', label: 'Batch Samples (n)', type: 'number', default: 32, min: 4, max: 256, step: 4 },
      { key: 'strategy', label: 'Sampling Strategy', type: 'select', default: 'min_p', options: ['min_p', 'top_p', 'top_k', 'typical', 'diverse_beam'] },
      { key: 'cfg_scale', label: 'CFG Scale w', type: 'number', default: 1.35, min: 0.5, max: 3.0, step: 0.05 },
      { key: 'temperature', label: 'Temperature (T)', type: 'number', default: 0.9, min: 0.5, max: 1.4, step: 0.05 },
      { key: 'perceiver_layers', label: 'Perceiver Layers', type: 'number', default: 2, min: 1, max: 4, step: 1 },
    ],
  },
  {
    blockType: 'pocket_ensemble_cfg',
    name: 'Pocket Ensemble CFG',
    shortName: 'Ensemble CFG',
    category: 'generators',
    pythonSymbol: 'geneuclidean.sampling.pocket_steering.sample_contrastive_ensemble_llm',
    executionTier: 'optional_torch',
    badge: 'ΔPMI 3D',
    summary:
      'Multi-conformation K-pocket logit ensembling with zero-retraining off-target repulsion and exact ΔPMI_3D specificity scoring.',
    inputs: [
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: true, multi: false },
      { id: 'off_pocket', label: 'off_pocket', signalType: 'POCKET_3D', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Steered Batch[n]' },
      { id: 'pmi_scores', label: 'pmi_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'PMI_3D / ΔPMI' },
    ],
    params: [
      {
        key: 'backbone_id',
        label: 'Generator Backbone',
        type: 'select',
        default: 'datamol-io/safe-gpt',
        options: [
          'datamol-io/safe-gpt',
          'ibm-research/GP-MoLFormer-Uniq',
          'chandar-lab/NovoMolGen_157M_SMILES_BPE',
        ],
      },
      { key: 'w_on', label: 'On-Target Guidance w_on', type: 'number', default: 1.6, min: 0.5, max: 3.0, step: 0.05 },
      { key: 'w_off', label: 'Off-Target Repulsion w_off', type: 'number', default: 0.65, min: 0.0, max: 2.0, step: 0.05 },
      { key: 'num_samples', label: 'Samples (n)', type: 'number', default: 32, min: 4, max: 128, step: 4 },
      { key: 'compute_pmi_3d', label: 'Compute ΔPMI_3D Selectivity', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'selectivity_mtl_aligner',
    name: 'Selectivity & MTL Aligner',
    shortName: 'MTL Aligner',
    category: 'generators',
    pythonSymbol: 'geneuclidean.training.mtl_intervention.GradientConflictMediator',
    executionTier: 'optional_torch',
    badge: 'AIM · GDPO',
    summary:
      'Fine-tunes contrastive paralog selectivity margins (DRD1 vs DRD2 AUROC 0.417 → 0.735) and resolves multi-task gradient conflicts via AIM/PCGrad/GDPO.',
    inputs: [
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: true, multi: false },
      { id: 'off_pocket', label: 'off_pocket', signalType: 'POCKET_3D', direction: 'input', required: false, multi: false },
      {
        id: 'feedback_scores',
        label: 'feedback',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'aligned_vec', label: 'aligned_vec', signalType: 'LATENT_VEC', direction: 'output', multi: true, shapeHint: 'Tuned Head' },
      { id: 'diagnostics', label: 'diagnostics', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'Grad Conflict' },
    ],
    params: [
      { key: 'margin', label: 'Paralog Margin m', type: 'number', default: 0.15, min: 0.05, max: 0.5, step: 0.05 },
      { key: 'selectivity_weight', label: 'Selectivity Weight', type: 'number', default: 0.5, min: 0.1, max: 2.0, step: 0.05 },
      { key: 'mtl_strategy', label: 'MTL Conflict Policy', type: 'select', default: 'aim_policy', options: ['aim_policy', 'pcgrad', 'scalarized'] },
      { key: 'gdpo_beta', label: 'GDPO KL β', type: 'number', default: 0.04, min: 0.01, max: 0.5, step: 0.01 },
      { key: 'steps', label: 'Tuning Steps', type: 'number', default: 30, min: 10, max: 200, step: 10 },
    ],
  },

  // =========================================================================
  // Category 4 — Virtual Screening & Retrieval (4 blocks)
  // =========================================================================
  {
    blockType: 'ligand_vector_index',
    name: 'Ligand Vector Index',
    shortName: 'Vector Index',
    category: 'screening',
    pythonSymbol: 'geneuclidean.screening.screener.LigandVectorIndex',
    executionTier: 'zero_torch',
    badge: 'FP16 BLAS',
    summary:
      'Streaming float16 L2-normalized vector index supporting chunked BLAS cosine retrieval over Parquet/CSV/SMILES catalogs in bounded memory.',
    inputs: [
      { id: 'embeddings', label: 'embeddings', signalType: 'LATENT_VEC', direction: 'input', required: true, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'index_vecs', label: 'index_vecs', signalType: 'LATENT_VEC', direction: 'output', multi: true, shapeHint: 'FP16 Index[N×D]' },
      { id: 'pool_mols', label: 'pool_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true },
    ],
    params: [
      { key: 'chunk_size', label: 'BLAS Chunk Size', type: 'number', default: 65536, min: 1024, max: 262144, step: 1024 },
      { key: 'dtype', label: 'Index Precision', type: 'select', default: 'float16', options: ['float16', 'float32'] },
      { key: 'mmap', label: 'Memory-Mapped Index', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'pocket_screener',
    name: 'Contrastive Pocket Screener',
    shortName: 'Pocket Screener',
    category: 'screening',
    pythonSymbol: 'geneuclidean.screening.screener.PocketScreener.screen',
    executionTier: 'optional_torch',
    badge: '0.02 ms/q',
    summary:
      'Millisecond (0.02–0.04 ms/query) multi-conformation 3D dense retrieval with optional off-target vector repulsion and MedChem triage.',
    inputs: [
      { id: 'on_pockets', label: 'on_pockets', signalType: 'POCKET_3D', direction: 'input', required: true, multi: false },
      { id: 'off_pocket', label: 'off_pocket', signalType: 'POCKET_3D', direction: 'input', required: false, multi: false },
      { id: 'index_vecs', label: 'index_vecs', signalType: 'LATENT_VEC', direction: 'input', required: true, multi: false },
      { id: 'pool_mols', label: 'pool_mols', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'hits', label: 'hits', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Top-K Hits' },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'CLIP + ΔpKd' },
    ],
    params: [
      { key: 'off_weight', label: 'Off-Target Repulsion λ', type: 'number', default: 0.5, min: 0.0, max: 1.5, step: 0.05 },
      { key: 'top_k', label: 'Top-K Hits', type: 'number', default: 50, min: 5, max: 500, step: 5 },
      { key: 'rerank_pkd', label: 'Stage-2 Neural pKd Rerank', type: 'boolean', default: true },
      { key: 'triage_filter', label: 'MedChem Hygiene Gate', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'neural_pkd_reranker',
    name: 'Neural pKd & MaxMin Reranker',
    shortName: 'pKd Reranker',
    category: 'screening',
    pythonSymbol: 'geneuclidean.sampling.rerank.rerank_and_select',
    executionTier: 'optional_torch',
    badge: 'MaxMin',
    summary:
      'Reranks candidate molecules by neural 3D predicted pKd / MPO desirability and enforces Bemis–Murcko diversity via greedy MaxMin Tanimoto.',
    inputs: [
      { id: 'pocket_graph', label: 'pocket_graph', signalType: 'SE3_GRAPH', direction: 'input', required: false, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'ranked_mols', label: 'ranked_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Diverse Top-K' },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'pKd + MPO' },
    ],
    params: [
      { key: 'top_k', label: 'Select Top-K', type: 'number', default: 16, min: 4, max: 128, step: 4 },
      { key: 'min_tanimoto_dist', label: 'Min Tanimoto Dist', type: 'number', default: 0.35, min: 0.1, max: 0.7, step: 0.05 },
      { key: 'quality_weight', label: 'Quality Weight', type: 'number', default: 0.7, min: 0.2, max: 1.0, step: 0.05 },
    ],
  },
  {
    blockType: 'catalog_grounding',
    name: 'Catalog Grounding',
    shortName: 'Catalog Grounding',
    category: 'screening',
    pythonSymbol: 'geneuclidean.screening.screener.PocketScreener.ground_to_catalog',
    executionTier: 'zero_torch',
    badge: 'Buyable',
    summary:
      'Maps de novo generated molecules onto nearest buyable/make-on-demand catalog analogs ranked jointly by 3D contrastive score and 2D Tanimoto.',
    inputs: [
      { id: 'query_mols', label: 'query_mols', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'index_vecs', label: 'index_vecs', signalType: 'LATENT_VEC', direction: 'input', required: true, multi: false },
      { id: 'on_pockets', label: 'on_pockets', signalType: 'POCKET_3D', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'grounded_mols', label: 'grounded_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Buyable Analogs' },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'Tanimoto + 3D' },
    ],
    params: [
      { key: 'top_k_per_query', label: 'Analogs / Query', type: 'number', default: 5, min: 1, max: 20, step: 1 },
      { key: 'alpha_3d', label: '3D vs 2D Blend α_3D', type: 'number', default: 0.5, min: 0.0, max: 1.0, step: 0.05 },
      { key: 'off_weight', label: 'Off-Target Repulsion', type: 'number', default: 0.5, min: 0.0, max: 1.5, step: 0.05 },
    ],
  },

  // =========================================================================
  // Category 5 — Active Learning DMTA (6 blocks, v5.1)
  // =========================================================================
  {
    blockType: 'tanimoto_gp',
    name: 'Tanimoto-Kernel GP',
    shortName: 'Tanimoto GP',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.TanimotoGP',
    executionTier: 'v51_al',
    badge: 'v5.1 · Exact',
    summary:
      'Exact Gaussian Process with analytical positive-semidefinite Tanimoto / MinMax fingerprint kernel and Cholesky closed-form posterior mean & variance.',
    inputs: [
      { id: 'pool_mols', label: 'pool_mols', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      {
        id: 'observations',
        label: 'observations',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'output', multi: true, shapeHint: 'GP(μ, σ²_exact)' },
    ],
    params: [
      { key: 'kernel', label: 'Fingerprint Kernel', type: 'select', default: 'tanimoto', options: ['tanimoto', 'minmax_count'] },
      { key: 'fp_bits', label: 'ECFP4 Bits', type: 'select', default: 2048, options: [1024, 2048] },
      { key: 'noise_var', label: 'Observation Noise σ²_n', type: 'number', default: 0.05, min: 0.001, max: 1.0, step: 0.01 },
      { key: 'signal_var', label: 'Kernel Variance σ²_f', type: 'number', default: 1.0, min: 0.1, max: 5.0, step: 0.1 },
      { key: 'cold_start', label: 'Cold-Start Seeding', type: 'select', default: 'butina_medoids', options: ['butina_medoids', 'maxmin', 'random'] },
    ],
  },
  {
    blockType: 'ensemble_surrogate',
    name: 'Ensemble Surrogate',
    shortName: 'Ensemble UQ',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.EnsembleSurrogate',
    executionTier: 'v51_al',
    badge: 'RF + HGB',
    summary:
      'Bootstrap RandomForest + HistGradientBoosting ensemble surrogate with explicit epistemic vs. aleatoric uncertainty decomposition.',
    inputs: [
      { id: 'pool_mols', label: 'pool_mols', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      {
        id: 'observations',
        label: 'observations',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'output', multi: true, shapeHint: 'Ens(μ, σ_ep, σ_al)' },
    ],
    params: [
      { key: 'n_estimators', label: 'Estimators', type: 'number', default: 64, min: 16, max: 256, step: 16 },
      { key: 'fp_bits', label: 'ECFP4 Bits', type: 'select', default: 1024, options: [1024, 2048] },
      { key: 'include_2d_desc', label: 'Append 12 RDKit Descriptors', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'conformal_calibrator',
    name: 'Split-Conformal UQ',
    shortName: 'Conformal UQ',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.surrogates.ConformalSurrogate',
    executionTier: 'v51_al',
    badge: '90% Cov.',
    summary:
      'Wraps any surrogate posterior in finite-sample Split-Conformal intervals supporting normalized and Tanimoto-AD covariate-shift reweighting.',
    inputs: [
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'input', required: true, multi: false },
      { id: 'calibration_scores', label: 'cal_scores', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'calibrated_uq', label: 'calibrated_uq', signalType: 'UNCERTAINTY_DIST', direction: 'output', multi: true, shapeHint: 'Conformal[lo, hi]' },
    ],
    params: [
      {
        key: 'mode',
        label: 'Conformal Mode',
        type: 'select',
        default: 'normalized',
        options: ['normalized', 'homoscedastic', 'weighted_covariate_shift'],
      },
      { key: 'alpha', label: 'Miscoverage α (90% = 0.10)', type: 'number', default: 0.1, min: 0.02, max: 0.3, step: 0.02 },
      { key: 'scaffold_grouped', label: 'Bemis–Murcko Cal Split', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'batch_acquisition',
    name: 'Batch Acquisition',
    shortName: 'Batch Acquisition',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.acquisitions.ACQUISITION_REGISTRY',
    executionTier: 'v51_al',
    badge: '12 Strategies',
    summary:
      'Selects diverse, uncertainty-aware batches across 12 strategies (KrigingBeliever, DPP-UCB, MaxMin-UCB, qEHVI, ParEGO, Constrained-EI) with async pending masks.',
    inputs: [
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'input', required: true, multi: false },
      { id: 'pool_mols', label: 'pool_mols', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'constraints', label: 'constraints', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: true },
    ],
    outputs: [
      { id: 'selected_batch', label: 'selected_batch', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Query Batch[B]' },
      { id: 'acq_scores', label: 'acq_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'Acq + Pareto HV' },
    ],
    params: [
      {
        key: 'strategy',
        label: 'Acquisition Function',
        type: 'select',
        default: 'kriging_believer',
        options: [
          'kriging_believer',
          'dpp_ucb',
          'maxmin_ucb',
          'qehvi',
          'pareto_scalarized_ucb',
          'log_ei',
          'constrained_ei',
          'avoidome_feasibility_ucb',
          'ucb',
          'ei',
          'pi',
          'thompson',
        ],
      },
      { key: 'batch_size', label: 'Batch Size (B)', type: 'number', default: 12, min: 2, max: 64, step: 2 },
      { key: 'ucb_beta', label: 'Exploration β', type: 'number', default: 1.5, min: 0.1, max: 4.0, step: 0.1 },
      { key: 'diversity_weight', label: 'Diversity Weight', type: 'number', default: 0.3, min: 0.0, max: 0.9, step: 0.05 },
      { key: 'mask_pending_async', label: 'Async In-Flight Batch Mask', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'multifidelity_router',
    name: 'Multi-Fidelity Router',
    shortName: 'MF-DMTA Router',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.multifidelity.MultiFidelityDMTA',
    executionTier: 'v51_al',
    badge: 'VoI · ρ≥0.3',
    summary:
      'Cost-aware Kennedy–O’Hagan residual GP router across Tier-0 (surrogate), Tier-1 (Vina/Audit), and Tier-2 (OpenFE/wet-lab) with Spearman negative-transfer guard.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'tier1_batch', label: 'tier1_batch', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Tier-1 Vina' },
      { id: 'tier2_batch', label: 'tier2_batch', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Tier-2 RBFE' },
      { id: 'routing_report', label: 'routing_report', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'Cost/VoI Ledger' },
    ],
    params: [
      { key: 'tier1_budget', label: 'Tier-1 Vina Quota', type: 'number', default: 12, min: 2, max: 64, step: 2 },
      { key: 'tier2_budget', label: 'Tier-2 RBFE/Assay Quota', type: 'number', default: 4, min: 1, max: 16, step: 1 },
      { key: 'cost_ratio_t2_t1', label: 'Cost Ratio (Tier2 / Tier1)', type: 'number', default: 20.0, min: 2.0, max: 100.0, step: 2.0 },
      { key: 'min_spearman_rho', label: 'Negative-Transfer Guardrail ρ', type: 'number', default: 0.3, min: 0.0, max: 0.8, step: 0.05 },
    ],
  },
  {
    blockType: 'combinatorial_ts',
    name: 'Combinatorial Thompson Sampler',
    shortName: 'Combinatorial TS',
    category: 'active_learning',
    pythonSymbol: 'geneuclidean.active_learning.CombinatorialThompsonSampler',
    executionTier: 'v51_al',
    badge: 'O(N_A+N_B)',
    summary:
      'Explores N_A × N_B reaction-grounded Enamine-style virtual spaces in O(N_A + N_B) oracle calls via reagent-level Bayesian Thompson posteriors.',
    inputs: [
      { id: 'reagents', label: 'reagents', signalType: 'REAGENTS', direction: 'input', required: true, multi: false },
      {
        id: 'feedback_scores',
        label: 'feedback',
        signalType: 'ORACLE_SCORES',
        direction: 'input',
        required: false,
        multi: false,
        acceptsFeedback: true,
      },
    ],
    outputs: [
      { id: 'sampled_mols', label: 'sampled_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Sampled Products' },
      { id: 'posterior', label: 'posterior', signalType: 'UNCERTAINTY_DIST', direction: 'output', multi: true, shapeHint: 'Reagent Posteriors' },
    ],
    params: [
      {
        key: 'reaction_name',
        label: 'Coupling Reaction',
        type: 'select',
        default: 'amide_coupling_acid_amine',
        options: [
          'amide_coupling_acid_amine',
          'sulfonamide_coupling',
          'reductive_amination_aldehyde',
          'suzuki_miyaura_biaryl',
          'buchwald_hartwig_amination',
          'snar_heteroaryl_fluoride',
          'cuaac_triazole_click',
          'urea_from_isocyanate',
        ],
      },
      { key: 'batch_size', label: 'Batch Size', type: 'number', default: 16, min: 4, max: 64, step: 4 },
      { key: 'warmup_per_reagent', label: 'Warm-up / Reagent', type: 'number', default: 2, min: 1, max: 5, step: 1 },
      { key: 'thermal_scale', label: 'Exploration Temp τ', type: 'number', default: 1.0, min: 0.2, max: 3.0, step: 0.1 },
      { key: 'triage_filter', label: 'MedChem Hygiene Gate', type: 'boolean', default: true },
    ],
  },

  // =========================================================================
  // Category 6 — 3D Physics & 7-Axis Pose Reality (4 blocks)
  // =========================================================================
  {
    blockType: 'protonation_ph74',
    name: 'pH 7.4 Protonation & Tautomer',
    shortName: 'pH 7.4 Protonation',
    category: 'physics_audit',
    pythonSymbol: 'geneuclidean.evaluation.protonation.protonate_smiles_ph74',
    executionTier: 'zero_torch',
    badge: 'pH 7.4',
    summary:
      'Ionizes basic amines / carboxylic acids at physiological pH 7.4 (critical for GPCR Asp3.32 salt bridges) and audits minor tautomer stability gaps.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'protonated_mols', label: 'protonated', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'pH 7.4 SMILES' },
      { id: 'microstate_report', label: 'report', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'Charge + Tautomer' },
    ],
    params: [
      { key: 'audit_tautomers', label: 'Audit Tautomer Microstates', type: 'boolean', default: true },
      { key: 'max_tautomers', label: 'Max Tautomers', type: 'number', default: 16, min: 4, max: 32, step: 4 },
      { key: 'tautomer_gap_threshold', label: 'Tautomer Gap Threshold', type: 'number', default: 2.0, min: 0.5, max: 5.0, step: 0.25 },
      { key: 'strict_input_charge', label: 'Strict Input Charge', type: 'boolean', default: false },
    ],
  },
  {
    blockType: 'autodock_vina',
    name: 'AutoDock Vina 3D',
    shortName: 'AutoDock Vina',
    category: 'physics_audit',
    pythonSymbol: 'geneuclidean.evaluation.docking_pipeline.dock_ligand_to_pocket',
    executionTier: 'zero_torch',
    badge: 'Vina 1.2.7',
    summary:
      'Generates ring-diverse ETKDGv3 conformers, docks into the orthosteric box with held-out AutoDock Vina v1.2.7, and profiles 6-ch PLIF anchors.',
    inputs: [
      { id: 'pocket', label: 'pocket', signalType: 'POCKET_3D', direction: 'input', required: true, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'poses', label: 'poses', signalType: 'POSE_3D', direction: 'output', multi: true, shapeHint: '3D Docked SDFs' },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'Vina ΔG, LE, PLIF' },
    ],
    params: [
      { key: 'exhaustiveness', label: 'Exhaustiveness', type: 'number', default: 8, min: 2, max: 32, step: 2 },
      { key: 'num_modes', label: 'Binding Modes', type: 'number', default: 9, min: 1, max: 20, step: 1 },
      { key: 'box_mode', label: 'Docking Box Mode', type: 'select', default: 'tight_per_axis', options: ['tight_per_axis', 'cubic_standard'] },
      { key: 'max_ring_confs', label: 'Ring-Diverse Confs', type: 'number', default: 4, min: 1, max: 8, step: 1 },
      { key: 'protonate_ph74', label: 'Ionize at pH 7.4', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'ring_torsion_strain',
    name: 'Cremer–Pople & Torsion Strain',
    shortName: 'Ring & Torsion',
    category: 'physics_audit',
    pythonSymbol: 'geneuclidean.evaluation.ring_strain.analyze_ring_strain',
    executionTier: 'zero_torch',
    badge: 'Beyond-PB',
    summary:
      'Catches the 39.3% of PoseBusters-passing poses hiding high-energy twist-boat saturated rings (Cremer–Pople MMFF94s) or strained conjugated torsions.',
    inputs: [
      { id: 'poses', label: 'poses', signalType: 'POSE_3D', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'clean_poses', label: 'clean_poses', signalType: 'POSE_3D', direction: 'output', multi: true, shapeHint: 'Low-Strain Poses' },
      { id: 'strain_scores', label: 'strain_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'Ring + Torsion kcal' },
    ],
    params: [
      { key: 'ring_strain_threshold_kcal', label: 'Ring Strain Max', type: 'number', default: 3.0, min: 1.0, max: 8.0, step: 0.5, unit: 'kcal/mol' },
      { key: 'ring_n_ref_confs', label: 'Ref Conformers', type: 'number', default: 24, min: 8, max: 48, step: 8 },
      { key: 'torsion_barrier_threshold_kcal', label: 'Torsion Barrier Max', type: 'number', default: 2.5, min: 1.0, max: 6.0, step: 0.25, unit: 'kcal/mol' },
      { key: 'torsion_max_red', label: 'Max Red Torsions', type: 'number', default: 0, min: 0, max: 3, step: 1 },
      { key: 'torsion_backend', label: 'Torsion Backend', type: 'select', default: 'smarts_mmff94s', options: ['smarts_mmff94s', 'roche_xml_plugin'] },
    ],
  },
  {
    blockType: 'pose_reality_auditor',
    name: '7-Axis PoseRealityAuditor',
    shortName: '7-Axis Auditor',
    category: 'physics_audit',
    pythonSymbol: 'geneuclidean.audit.pose_auditor.PoseRealityAuditor',
    executionTier: 'zero_torch',
    badge: '7-Axis QA',
    summary:
      'Unified Beyond-PoseBusters 7-axis physical & MedChem auditor: geometry/clashes, Cremer–Pople ring strain, torsions, pH 7.4 microstates, PLIF, hygiene & profile.',
    inputs: [
      { id: 'poses', label: 'poses', signalType: 'POSE_3D', direction: 'input', required: true, multi: false },
      { id: 'pocket', label: 'pocket', signalType: 'POCKET_3D', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'verified_poses', label: 'verified_poses', signalType: 'POSE_3D', direction: 'output', multi: true, shapeHint: '7-Axis Poses' },
      { id: 'reality_scores', label: 'reality_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'reality [0–1]' },
      { id: 'audit_report', label: 'audit_report', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: '7-Axis Report' },
    ],
    params: [
      { key: 'property_profile', label: 'Property Profile', type: 'select', default: 'cns_mpo_v2', options: ['cns_mpo_v2', 'oral_ro5', 'kinase_selectivity', 'beyond_ro5'] },
      { key: 'ring_strain_threshold_kcal', label: 'Ring Strain Max', type: 'number', default: 3.0, min: 1.0, max: 6.0, step: 0.5, unit: 'kcal/mol' },
      { key: 'torsion_barrier_threshold_kcal', label: 'Torsion Max', type: 'number', default: 2.5, min: 1.0, max: 6.0, step: 0.25, unit: 'kcal/mol' },
      { key: 'torsion_max_red', label: 'Max Red Torsions', type: 'number', default: 0, min: 0, max: 2, step: 1 },
      { key: 'strict_input_charge', label: 'Strict Input Charge', type: 'boolean', default: false },
    ],
  },

  // =========================================================================
  // Category 7 — Lead-Opt, ADMET & Synthesis (4 blocks)
  // =========================================================================
  {
    blockType: 'reaction_enumerator',
    name: '16-Reaction Makeable & Retro',
    shortName: '16-Rxn Synthesis',
    category: 'leadopt_admet',
    pythonSymbol: 'geneuclidean.synthesis.reactions.enumerate_makeable_analogs',
    executionTier: 'zero_torch',
    badge: 'Makeable',
    summary:
      'Enumerates 1-step parallelizable MedChem analogs across 16 Hartenfeller / Enamine REAL SMIRKS couplings with full reactant provenance & 1-step retro.',
    inputs: [
      { id: 'reagents', label: 'reagents', signalType: 'REAGENTS', direction: 'input', required: false, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'makeable_mols', label: 'makeable_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Makeable Analogs' },
      { id: 'synth_report', label: 'synth_report', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'SMIRKS + Retro' },
    ],
    params: [
      { key: 'mode', label: 'Synthesis Mode', type: 'select', default: 'enumerate_and_retro', options: ['enumerate_and_retro', 'enumerate_only', 'retrosynthesis_only'] },
      {
        key: 'reaction_family',
        label: 'Reaction SMIRKS',
        type: 'select',
        default: 'all_16',
        options: [
          'all_16',
          'amide_coupling_acid_amine',
          'sulfonamide_coupling',
          'reductive_amination_aldehyde',
          'suzuki_miyaura_biaryl',
          'buchwald_hartwig_amination',
          'snar_heteroaryl_fluoride',
          'cuaac_triazole_click',
        ],
      },
      { key: 'max_products', label: 'Max Products', type: 'number', default: 64, min: 8, max: 500, step: 8 },
      { key: 'triage_filter', label: 'Quarantine DCU/PAINS', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'leadopt_mmp_sar',
    name: 'Avoid-ome MMP & R-Group Cliffs',
    shortName: 'Avoid-ome MMP',
    category: 'leadopt_admet',
    pythonSymbol: 'geneuclidean.leadopt.mmp.AvoidomeMMPEngine',
    executionTier: 'zero_torch',
    badge: '1-Cut SAR',
    summary:
      'Runs Roche mmpdb-inspired 1-cut bioisostere rescues (15 curated hERG/MDR1/5-HT2B transforms like phenyl→pyridine) & MCS R-group activity cliffs.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'rescued_mols', label: 'rescued_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'MMP Rescues' },
      { id: 'mmp_scores', label: 'mmp_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'ΔhERG, ΔMPO' },
      { id: 'sar_report', label: 'sar_report', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'R-Group Cliffs' },
    ],
    params: [
      { key: 'mode', label: 'Lead-Opt Mode', type: 'select', default: 'mmp_and_cliffs', options: ['mmp_and_cliffs', 'mmp_rescue_only', 'rgroup_cliffs_only'] },
      { key: 'property_profile', label: 'Property Profile', type: 'select', default: 'cns_mpo_v2', options: ['cns_mpo_v2', 'oral_ro5', 'kinase_selectivity', 'beyond_ro5'] },
      { key: 'max_proposals', label: 'Max MMP Proposals', type: 'number', default: 20, min: 5, max: 50, step: 5 },
      { key: 'cliff_threshold', label: 'Activity Cliff Δ', type: 'number', default: 1.0, min: 0.3, max: 3.0, step: 0.1 },
      { key: 'triage_filter', label: 'MedChem Hygiene Gate', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'cns_admet_suite',
    name: '12-Ep Avoid-ome & Property Profile',
    shortName: '12-Ep Avoid-ome',
    category: 'leadopt_admet',
    pythonSymbol: 'geneuclidean.cns.admet.CNSAdmetSuite',
    executionTier: 'zero_torch',
    badge: '12-Ep · MPO',
    summary:
      'Scores all 12 industrial scaffold-CV Avoid-ome/DMPK endpoints (BBB, P-gp, hERG, MDR1, HLM, CYPs, 5-HT2B, H1, DILI, AMES, M3) + property profiles.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: true },
    ],
    outputs: [
      { id: 'safe_mols', label: 'safe_mols', signalType: 'MOL_BATCH', direction: 'output', multi: true, shapeHint: 'Safety-Gated' },
      { id: 'admet_scores', label: 'admet_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: '12-Ep + MPO' },
    ],
    params: [
      { key: 'property_profile', label: 'Property Profile', type: 'select', default: 'cns_mpo_v2', options: ['cns_mpo_v2', 'oral_ro5', 'kinase_selectivity', 'beyond_ro5'] },
      { key: 'admet_profile', label: 'Oracle Profile', type: 'select', default: 'v4_12ep', options: ['v4_12ep', 'v3_core3', 'profile_only'] },
      { key: 'min_mpo_v2', label: 'Min CNS MPO v2', type: 'number', default: 4.0, min: 2.0, max: 5.8, step: 0.2 },
      { key: 'max_herg_prob', label: 'Max P(hERG)', type: 'number', default: 0.3, min: 0.05, max: 0.8, step: 0.05 },
      { key: 'require_triage_pass', label: 'Enforce PAINS/DCU Quarantine', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'conformal_dmpk',
    name: 'Conformal DMPK Regressor',
    shortName: 'Conformal DMPK',
    category: 'leadopt_admet',
    pythonSymbol: 'geneuclidean.cns.dmpk_regression.ConformalDMPKRegressor',
    executionTier: 'zero_torch',
    badge: 'OpenADMET',
    summary:
      'OpenADMET-ExpansionRx & Biogen continuous pharmacokinetic regression (LogD, KSOL, HLM CLint, Solubility) with 90% Split-Conformal intervals.',
    inputs: [
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: true, multi: false },
    ],
    outputs: [
      { id: 'dmpk_intervals', label: 'intervals', signalType: 'UNCERTAINTY_DIST', direction: 'output', multi: true, shapeHint: 'ŷ ± q̂ [lo, hi]' },
      { id: 'dmpk_scores', label: 'dmpk_scores', signalType: 'ORACLE_SCORES', direction: 'output', multi: true, shapeHint: 'LogD / KSOL' },
    ],
    params: [
      {
        key: 'endpoint',
        label: 'DMPK Endpoint',
        type: 'select',
        default: 'OpenADMET_LogD',
        options: [
          'OpenADMET_LogD',
          'OpenADMET_log10_KSOL',
          'OpenADMET_log10_HLM_CLint',
          'Biogen_log_HLM_CLint',
          'Biogen_log_Solubility_pH68',
        ],
      },
      { key: 'alpha', label: 'Miscoverage α', type: 'number', default: 0.1, min: 0.05, max: 0.25, step: 0.05 },
      { key: 'n_estimators', label: 'Estimators', type: 'number', default: 80, min: 20, max: 160, step: 20 },
    ],
  },

  // =========================================================================
  // Category 8 — Outputs & Open-Pharma Export (4 blocks)
  // =========================================================================
  {
    blockType: 'openfe_rbfe_exporter',
    name: 'OpenFE RBFE Exporter',
    shortName: 'OpenFE RBFE',
    category: 'outputs',
    pythonSymbol: 'geneuclidean.interop.openfe_export.export_openfe_rbfe_network',
    executionTier: 'zero_torch',
    badge: 'Δq = 0',
    summary:
      'Exports a charge-consistent (Δq = 0), ring-strain-clean, MCS-aligned Lomap RBFE perturbation graph (JSON + SDF + OpenFE driver) with cycle closures.',
    inputs: [
      { id: 'pocket', label: 'pocket', signalType: 'POCKET_3D', direction: 'input', required: true, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: false, multi: false },
      { id: 'poses', label: 'poses', signalType: 'POSE_3D', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'rbfe_network', label: 'rbfe_network', signalType: 'RBFE_GRAPH', direction: 'output', multi: true, shapeHint: 'Lomap RBFE Graph' },
    ],
    params: [
      { key: 'require_same_charge', label: 'Enforce Δq = 0 Guard', type: 'boolean', default: true },
      { key: 'filter_ring_strain', label: 'RingStrainFilter Pre-Check', type: 'boolean', default: true },
      { key: 'max_ring_strain_kcal', label: 'Max Ring Strain', type: 'number', default: 3.0, min: 1.0, max: 6.0, step: 0.5, unit: 'kcal/mol' },
      { key: 'add_cycle_closures', label: 'Add Hysteresis Cycle Closures', type: 'boolean', default: true },
      { key: 'out_dir', label: 'Export Directory', type: 'string', default: 'results/openfe_circuit_export' },
    ],
  },
  {
    blockType: 'studio_3d_inspector',
    name: '3D Studio Inspector',
    shortName: '3D Studio Sink',
    category: 'outputs',
    pythonSymbol: 'geneuclidean.sdk.viz.visualize_complex',
    executionTier: 'zero_torch',
    badge: '3Dmol.js',
    summary:
      'Renders interactive 3Dmol.js WebGL protein-ligand complexes with 6-ch PLIF contacts and residue attention, with 1-click jump to 3D Studio.',
    inputs: [
      { id: 'pocket', label: 'pocket', signalType: 'POCKET_3D', direction: 'input', required: true, multi: false },
      { id: 'poses', label: 'poses', signalType: 'POSE_3D', direction: 'input', required: false, multi: false },
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: false },
      { id: 'audit_report', label: 'audit_report', signalType: 'PROVENANCE_REPORT', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'viz_manifest', label: 'viz_manifest', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: '3D WebGL Spec' },
    ],
    params: [
      { key: 'theme', label: 'Viewer Theme', type: 'select', default: 'dark', options: ['dark', 'light'] },
      { key: 'show_native_ghost', label: 'Overlay Native Co-Crystal', type: 'boolean', default: true },
      { key: 'highlight_anchors', label: 'Highlight Canonical Anchors', type: 'boolean', default: true },
    ],
  },
  {
    blockType: 'mcp_tool_export',
    name: 'Agentic MCP Tool Server',
    shortName: 'MCP Tool Server',
    category: 'outputs',
    pythonSymbol: 'geneuclidean.mcp_server.invoke_dmta_tool',
    executionTier: 'zero_torch',
    badge: 'SHA-256',
    summary:
      'Packages pipeline endpoints as deterministic, SHA-256 provenance-stamped Model Context Protocol (MCP) tools for autonomous pharma DMTA agents.',
    inputs: [
      { id: 'audit_report', label: 'audit_report', signalType: 'PROVENANCE_REPORT', direction: 'input', required: false, multi: true },
      { id: 'uncertainty', label: 'uncertainty', signalType: 'UNCERTAINTY_DIST', direction: 'input', required: false, multi: false },
      { id: 'molecules', label: 'molecules', signalType: 'MOL_BATCH', direction: 'input', required: false, multi: false },
    ],
    outputs: [
      { id: 'mcp_manifest', label: 'mcp_manifest', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: '6 Typed MCP Tools' },
    ],
    params: [
      {
        key: 'tool_name',
        label: 'Primary MCP Tool',
        type: 'select',
        default: 'audit_complex',
        options: [
          'score_molecules',
          'evaluate_property_profile',
          'audit_complex',
          'enumerate_makeable_analogs',
          'propose_mmp_rescues',
          'match_retrosynthesis',
        ],
      },
      { key: 'transport', label: 'Transport', type: 'select', default: 'python_direct', options: ['python_direct', 'mcp_stdio'] },
    ],
  },
  {
    blockType: 'dmta_ledger',
    name: 'DMTA Ledger & Stopping Advisor',
    shortName: 'DMTA Ledger',
    category: 'outputs',
    pythonSymbol: 'geneuclidean.sdk.credibility.build_credibility_dossier',
    executionTier: 'zero_torch',
    badge: 'FDA/EMA 10/10',
    summary:
      'Seals every circuit execution or multi-round DMTA loop into a chained SHA-256 DMTALedger + FDA/EMA CredibilityDossier + StoppingAdvisor.',
    inputs: [
      { id: 'scores', label: 'scores', signalType: 'ORACLE_SCORES', direction: 'input', required: false, multi: true },
      { id: 'uncertainty', label: 'uncertainty', signalType: 'UNCERTAINTY_DIST', direction: 'input', required: false, multi: false },
      { id: 'reports', label: 'reports', signalType: 'PROVENANCE_REPORT', direction: 'input', required: false, multi: true },
    ],
    outputs: [
      { id: 'dossier', label: 'dossier', signalType: 'PROVENANCE_REPORT', direction: 'output', multi: true, shapeHint: 'Ledger + Verdict' },
    ],
    params: [
      { key: 'ei_saturation_tol', label: 'EI Saturation Threshold', type: 'number', default: 0.02, min: 0.005, max: 0.1, step: 0.005 },
      { key: 'jaccard_stability_tol', label: 'Top-K Jaccard Stop', type: 'number', default: 0.85, min: 0.5, max: 0.98, step: 0.05 },
      { key: 'export_fda_ema_dossier', label: 'Generate FDA/EMA 10/10 Dossier', type: 'boolean', default: true },
    ],
  },
];

export const BLOCK_CATALOG_MAP: Record<string, BlockDefinition> = Object.fromEntries(
  BLOCK_CATALOG.map((b) => [b.blockType, b]),
);

export function createDefaultParams(blockType: string): Record<string, number | string | boolean> {
  const def = BLOCK_CATALOG_MAP[blockType];
  if (!def) return {};
  const params: Record<string, number | string | boolean> = {};
  for (const p of def.params) {
    params[p.key] = p.default;
  }
  return params;
}

export function getWiringRemediationHint(source: SignalType, target: SignalType): string {
  const key = `${source}->${target}`;
  const hints: Record<string, string> = {
    'POCKET_3D->SE3_GRAPH':
      'Insert PaiNN-SE(3), EGNN-3D, or E3NN Irreps to encode the 3D pocket into an equivariant graph.',
    'MOL_BATCH->POSE_3D':
      'Insert AutoDock Vina 3D (or Boltz-2 / AF3 Co-Fold) to generate 3D ligand poses before geometry auditing.',
    'MOL_BATCH->LATENT_VEC':
      'Insert DrugCLIP Encoder or IBM MMELON (84M) to embed SMILES into L2-normalized vectors.',
    'REAGENTS->MOL_BATCH':
      'Insert 16-Reaction Makeable & Retro or Combinatorial Thompson Sampler to couple reagents into products.',
    'ORACLE_SCORES->UNCERTAINTY_DIST':
      'Insert Tanimoto-Kernel GP or Ensemble Surrogate to fit a predictive posterior with epistemic uncertainty.',
    'POSE_3D->RBFE_GRAPH':
      'Wire POSE_3D or MOL_BATCH into OpenFE RBFE Exporter to build a charge-matched (Δq = 0) Lomap graph.',
  };
  if (hints[key]) return hints[key];
  const srcLabel = SIGNAL_TYPE_META[source]?.label ?? source;
  const dstLabel = SIGNAL_TYPE_META[target]?.label ?? target;
  return `Signal mismatch: ${srcLabel} cannot connect to ${dstLabel}. Connect to pulsing ${srcLabel} sockets.`;
}

export function formatNodeParamSummary(node: CircuitNode): string {
  const p = node.params;
  switch (node.blockType) {
    case 'protein_pocket': {
      const tid = String(p.target_id ?? 'DRD1');
      const pdb = TARGET_PDB_MAP[tid] ?? 'PDB';
      return `${tid} · ${pdb} · ${Number(p.pocket_cutoff ?? 10).toFixed(0)}Å`;
    }
    case 'multiconformer_ensemble':
      return `${p.on_target_id ?? 'DRD1'}(${p.n_conformers ?? 2}×) vs ${p.off_target_id ?? 'DRD2'}`;
    case 'reagent_library':
      return `${String(p.reagent_pack ?? 'cns_amines').replace('_', ' ')} · N=${p.max_reagents ?? 32}`;
    case 'boltz2_cofold':
      return String(p.structure_source ?? 'ACHE_donepezil').replace('_cofold.cif', '');
    case 'chembl_pool':
      return `${String(p.dataset ?? 'DRD1').split('_')[0]} · N=${p.pool_size ?? 400}`;
    case 'painn_se3':
      return `${p.num_layers ?? 4}L · d=${p.hidden_dim ?? 192} · ${p.cutoff ?? 5.5}Å`;
    case 'egnn_3d':
      return `${p.num_layers ?? 4}L · d=${p.hidden_dim ?? 192} · ${p.cutoff ?? 5.5}Å`;
    case 'e3nn_irreps':
      return `${p.num_layers ?? 4}L · l≤${p.e3nn_lmax ?? 1} · d=${p.hidden_dim ?? 192}`;
    case 'drugclip_encoder':
      return `${p.backend === 'ecfp4_binary' ? 'ECFP4' : 'CLIP'} · ${p.clip_dim ?? 128}D`;
    case 'ibm_mmelon':
      return `84M · 768D · ${String(p.device ?? 'auto')}`;
    case 'safe_gpt':
      return `${p.mode === 'decorate_scaffold' ? 'Decor' : 'DeNovo'} · n=${p.num_samples ?? 32} · w=${Number(p.cfg_scale ?? 1.6).toFixed(1)}`;
    case 'chemical_llm': {
      const shortModel = String(p.backbone_id ?? 'GP-MoLFormer').split('/').pop() ?? 'LLM';
      return `${shortModel.slice(0, 11)} · w=${Number(p.cfg_scale ?? 1.35).toFixed(1)}`;
    }
    case 'pocket_ensemble_cfg':
      return `w+=${Number(p.w_on ?? 1.6).toFixed(1)} · w−=${Number(p.w_off ?? 0.65).toFixed(2)}`;
    case 'selectivity_mtl_aligner':
      return `${String(p.mtl_strategy ?? 'aim').split('_')[0]} · m=${p.margin ?? 0.15}`;
    case 'ligand_vector_index':
      return `${p.dtype ?? 'fp16'} · ${Math.round(Number(p.chunk_size ?? 65536) / 1024)}k ch`;
    case 'pocket_screener':
      return `Top-${p.top_k ?? 50} · λ=${Number(p.off_weight ?? 0.5).toFixed(2)}`;
    case 'neural_pkd_reranker':
      return `Top-${p.top_k ?? 16} · d≥${p.min_tanimoto_dist ?? 0.35}`;
    case 'catalog_grounding':
      return `k=${p.top_k_per_query ?? 5} · α=${Number(p.alpha_3d ?? 0.5).toFixed(2)}`;
    case 'tanimoto_gp':
      return `${p.kernel === 'minmax_count' ? 'MinMax' : 'Tanimoto'} · ${p.fp_bits ?? 2048}b`;
    case 'ensemble_surrogate':
      return `RF+HGB · ${p.n_estimators ?? 64}t`;
    case 'conformal_calibrator': {
      const m =
        p.mode === 'weighted_covariate_shift'
          ? 'CovShift'
          : p.mode === 'homoscedastic'
            ? 'Homo'
            : 'Norm';
      return `${m} · ${Math.round((1 - Number(p.alpha ?? 0.1)) * 100)}%`;
    }
    case 'batch_acquisition': {
      const map: Record<string, string> = {
        kriging_believer: 'KB',
        dpp_ucb: 'DPP-UCB',
        maxmin_ucb: 'MaxMin',
        qehvi: 'qEHVI',
        pareto_scalarized_ucb: 'ParEGO',
        log_ei: 'LogEI',
        constrained_ei: 'cEI',
        avoidome_feasibility_ucb: 'AvUCB',
        ucb: 'UCB',
        ei: 'EI',
        pi: 'PI',
        thompson: 'TS',
      };
      const label = map[String(p.strategy ?? 'kriging_believer')] ?? String(p.strategy);
      return `${label} · B=${p.batch_size ?? 12} · β=${p.ucb_beta ?? 1.5}`;
    }
    case 'multifidelity_router':
      return `T1=${p.tier1_budget ?? 12} · T2=${p.tier2_budget ?? 4} · ρ≥${p.min_spearman_rho ?? 0.3}`;
    case 'combinatorial_ts':
      return `${String(p.reaction_name ?? 'amide').split('_')[0]} · B=${p.batch_size ?? 16}`;
    case 'protonation_ph74':
      return `pH 7.4${p.audit_tautomers ? ` · ≤${p.max_tautomers ?? 16}t` : ''}`;
    case 'autodock_vina':
      return `exh=${p.exhaustiveness ?? 8} · ${p.num_modes ?? 9}m · pH7.4`;
    case 'ring_torsion_strain':
      return `Ring≤${p.ring_strain_threshold_kcal ?? 3.0} · Tor≤${p.torsion_barrier_threshold_kcal ?? 2.5}`;
    case 'pose_reality_auditor':
      return `${String(p.property_profile ?? 'cns_mpo_v2').replace('_v2', '')} · ≤${p.ring_strain_threshold_kcal ?? 3.0}k`;
    case 'reaction_enumerator':
      return `${String(p.reaction_family ?? 'all_16').split('_')[0]} · ≤${p.max_products ?? 64}`;
    case 'leadopt_mmp_sar':
      return `1-Cut · ≤${p.max_proposals ?? 20} · Δ≥${p.cliff_threshold ?? 1.0}`;
    case 'cns_admet_suite':
      return `12-Ep · MPO≥${p.min_mpo_v2 ?? 4.0}`;
    case 'conformal_dmpk':
      return `${String(p.endpoint ?? 'LogD').replace('OpenADMET_', '').replace('Biogen_', '')} · ${Math.round((1 - Number(p.alpha ?? 0.1)) * 100)}%`;
    case 'openfe_rbfe_exporter':
      return `${p.require_same_charge ? 'Δq=0 · ' : ''}${p.add_cycle_closures ? 'Cycle' : 'Star'} Lomap`;
    case 'studio_3d_inspector':
      return `3Dmol · ${p.highlight_anchors ? 'Anchors' : 'Pocket'}`;
    case 'mcp_tool_export':
      return `${String(p.tool_name ?? 'audit_complex').replace('_', ' ')}`;
    case 'dmta_ledger':
      return `EI≤${p.ei_saturation_tol ?? 0.02} · J≥${p.jaccard_stability_tol ?? 0.85}`;
    default:
      return node.blockType;
  }
}

// ===========================================================================
// 5 Flagship One-Click End-to-End Presets (exact CircuitGraphV1 contract)
// ===========================================================================

export const CIRCUIT_PRESETS: CircuitPreset[] = [
  {
    id: 'se3_denovo_cns',
    shortLabel: 'SE(3) De Novo',
    name: 'SE(3) Pocket-Conditioned De Novo Circuit',
    tagline:
      'Tourette DRD1 (9LLG) chiral PaiNN-SE(3) → SAFE-GPT-87M → 12-Ep Avoid-ome + AutoDock Vina → 7-Axis PoseRealityAuditor + OpenFE RBFE (Δq=0).',
    badge: 'Flagship',
    graph: {
      schemaVersion: SCHEMA_VERSION,
      id: 'preset_se3_denovo_cns',
      name: 'SE(3) Pocket-Conditioned De Novo Circuit',
      description:
        'End-to-end Tourette DRD1 (9LLG) anti-EPS de novo discovery uniting chiral PaiNN-SE(3), SAFE-GPT-87M, 12-endpoint Avoid-ome, held-out AutoDock Vina v1.2.7, 7-Axis PoseRealityAuditor, and charge-matched OpenFE RBFE.',
      metadata: {
        presetId: 'se3_denovo_cns',
        author: 'GenEuclidean v5.1',
        tags: ['SE(3)', 'SAFE-GPT', 'Vina', '7-Axis', 'OpenFE'],
      },
      execution: {
        mode: 'browser_sim',
        seed: 42,
        dmtaRounds: 1,
        targetContext: 'DRD1',
        budget: 32,
      },
      nodes: [
        // Row 1: 3D Pocket -> SE(3) Encoding -> SAFE-GPT De Novo -> 12-Ep Avoid-ome
        {
          id: 'n_pocket',
          blockType: 'protein_pocket',
          position: { x: 32, y: 40 },
          params: { target_id: 'DRD1', pocket_cutoff: 10.0, max_atoms: 384, keep_cofactors: true },
        },
        {
          id: 'n_painn',
          blockType: 'painn_se3',
          position: { x: 324, y: 40 },
          params: {
            num_layers: 4,
            hidden_dim: 192,
            vector_dim: 64,
            out_dim: 256,
            num_rbf: 24,
            cutoff: 5.5,
            max_neighbors: 24,
            use_se3_chirality: true,
          },
        },
        {
          id: 'n_safegpt',
          blockType: 'safe_gpt',
          position: { x: 616, y: 40 },
          params: {
            mode: 'de_novo',
            num_samples: 32,
            strategy: 'min_p',
            cfg_scale: 1.6,
            temperature: 0.95,
            min_p: 0.08,
            num_queries: 16,
            core_smiles_with_dummy: '',
          },
        },
        {
          id: 'n_admet',
          blockType: 'cns_admet_suite',
          position: { x: 908, y: 40 },
          params: {
            property_profile: 'cns_mpo_v2',
            admet_profile: 'v4_12ep',
            min_mpo_v2: 4.5,
            max_herg_prob: 0.3,
            require_triage_pass: true,
          },
        },
        // Row 2: AutoDock Vina 3D -> 7-Axis PoseRealityAuditor -> OpenFE RBFE & 3D Studio
        {
          id: 'n_vina',
          blockType: 'autodock_vina',
          position: { x: 32, y: 316 },
          params: {
            exhaustiveness: 8,
            num_modes: 9,
            box_mode: 'tight_per_axis',
            max_ring_confs: 4,
            protonate_ph74: true,
          },
        },
        {
          id: 'n_audit',
          blockType: 'pose_reality_auditor',
          position: { x: 324, y: 316 },
          params: {
            property_profile: 'cns_mpo_v2',
            ring_strain_threshold_kcal: 3.0,
            torsion_barrier_threshold_kcal: 2.5,
            torsion_max_red: 0,
            strict_input_charge: false,
          },
        },
        {
          id: 'n_openfe',
          blockType: 'openfe_rbfe_exporter',
          position: { x: 616, y: 316 },
          params: {
            require_same_charge: true,
            filter_ring_strain: true,
            max_ring_strain_kcal: 3.0,
            add_cycle_closures: true,
            out_dir: 'results/openfe_drd1_denovo',
          },
        },
        {
          id: 'n_studio3d',
          blockType: 'studio_3d_inspector',
          position: { x: 908, y: 316 },
          params: { theme: 'dark', show_native_ghost: true, highlight_anchors: true },
        },
      ],
      edges: [
        { id: 'e1', source: 'n_pocket', sourcePort: 'pocket', target: 'n_painn', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e2', source: 'n_painn', sourcePort: 'pocket_graph', target: 'n_safegpt', targetPort: 'pocket_graph', signalType: 'SE3_GRAPH', kind: 'forward' },
        { id: 'e3', source: 'n_safegpt', sourcePort: 'molecules', target: 'n_admet', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e4', source: 'n_safegpt', sourcePort: 'molecules', target: 'n_vina', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e5', source: 'n_pocket', sourcePort: 'pocket', target: 'n_vina', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e6', source: 'n_admet', sourcePort: 'safe_mols', target: 'n_openfe', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e7', source: 'n_pocket', sourcePort: 'pocket', target: 'n_openfe', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e8', source: 'n_vina', sourcePort: 'poses', target: 'n_audit', targetPort: 'poses', signalType: 'POSE_3D', kind: 'forward' },
        { id: 'e9', source: 'n_pocket', sourcePort: 'pocket', target: 'n_audit', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e10', source: 'n_pocket', sourcePort: 'pocket', target: 'n_studio3d', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e11', source: 'n_audit', sourcePort: 'verified_poses', target: 'n_studio3d', targetPort: 'poses', signalType: 'POSE_3D', kind: 'forward' },
        { id: 'e12', source: 'n_admet', sourcePort: 'admet_scores', target: 'n_studio3d', targetPort: 'scores', signalType: 'ORACLE_SCORES', kind: 'forward' },
        { id: 'e13', source: 'n_audit', sourcePort: 'audit_report', target: 'n_studio3d', targetPort: 'audit_report', signalType: 'PROVENANCE_REPORT', kind: 'forward' },
      ],
    },
  },
  {
    id: 'al_dmta_closed_loop',
    shortLabel: 'AL DMTA Loop',
    name: 'Multi-Fidelity Active Learning DMTA Closed Loop',
    tagline:
      '5-round Lab-in-a-Loop over 400 Murcko DRD1 scaffolds: TanimotoGP → Split-Conformal UQ → KrigingBeliever → Multi-Fidelity Router → Vina with live DMTA Feedback Loop.',
    badge: '↻ 5 Rounds',
    graph: {
      schemaVersion: SCHEMA_VERSION,
      id: 'preset_al_dmta_closed_loop',
      name: 'Multi-Fidelity Active Learning DMTA Closed Loop',
      description:
        'Closed-loop Active Learning DMTA over 400 Bemis–Murcko DRD1 scaffolds uniting exact TanimotoGP, Split-Conformal UQ, KrigingBeliever batch acquisition, Multi-Fidelity VoI routing, and a live DMTA Feedback Loop.',
      metadata: {
        presetId: 'al_dmta_closed_loop',
        author: 'GenEuclidean v5.1',
        tags: ['Active Learning', 'TanimotoGP', 'Conformal', 'DMTA Loop'],
      },
      execution: {
        mode: 'browser_sim',
        seed: 42,
        dmtaRounds: 5,
        targetContext: 'DRD1',
        budget: 48,
      },
      nodes: [
        // Row 1: ChEMBL Pool -> Tanimoto GP -> Conformal UQ -> Batch Acquisition
        {
          id: 'n_pool',
          blockType: 'chembl_pool',
          position: { x: 32, y: 84 },
          params: { dataset: 'DRD1_chembl37_400', pool_size: 400, dedup_murcko: true },
        },
        {
          id: 'n_gp',
          blockType: 'tanimoto_gp',
          position: { x: 324, y: 84 },
          params: {
            kernel: 'tanimoto',
            fp_bits: 2048,
            noise_var: 0.05,
            signal_var: 1.0,
            cold_start: 'butina_medoids',
          },
        },
        {
          id: 'n_conf',
          blockType: 'conformal_calibrator',
          position: { x: 616, y: 84 },
          params: { mode: 'normalized', alpha: 0.1, scaffold_grouped: true },
        },
        {
          id: 'n_acq',
          blockType: 'batch_acquisition',
          position: { x: 908, y: 84 },
          params: {
            strategy: 'kriging_believer',
            batch_size: 12,
            ucb_beta: 1.5,
            diversity_weight: 0.3,
            mask_pending_async: true,
          },
        },
        // Row 2: Protein Pocket -> Multi-Fidelity Router -> AutoDock Vina -> DMTA Ledger
        {
          id: 'n_pocket',
          blockType: 'protein_pocket',
          position: { x: 32, y: 346 },
          params: { target_id: 'DRD1', pocket_cutoff: 10.0, max_atoms: 384, keep_cofactors: true },
        },
        {
          id: 'n_router',
          blockType: 'multifidelity_router',
          position: { x: 324, y: 346 },
          params: {
            tier1_budget: 12,
            tier2_budget: 4,
            cost_ratio_t2_t1: 20.0,
            min_spearman_rho: 0.3,
          },
        },
        {
          id: 'n_vina',
          blockType: 'autodock_vina',
          position: { x: 616, y: 346 },
          params: {
            exhaustiveness: 8,
            num_modes: 9,
            box_mode: 'tight_per_axis',
            max_ring_confs: 4,
            protonate_ph74: true,
          },
        },
        {
          id: 'n_ledger',
          blockType: 'dmta_ledger',
          position: { x: 908, y: 346 },
          params: {
            ei_saturation_tol: 0.02,
            jaccard_stability_tol: 0.85,
            export_fda_ema_dossier: true,
          },
        },
      ],
      edges: [
        { id: 'e1', source: 'n_pool', sourcePort: 'molecules', target: 'n_gp', targetPort: 'pool_mols', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e2', source: 'n_pool', sourcePort: 'molecules', target: 'n_acq', targetPort: 'pool_mols', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e3', source: 'n_gp', sourcePort: 'posterior', target: 'n_conf', targetPort: 'posterior', signalType: 'UNCERTAINTY_DIST', kind: 'forward' },
        { id: 'e4', source: 'n_conf', sourcePort: 'calibrated_uq', target: 'n_acq', targetPort: 'posterior', signalType: 'UNCERTAINTY_DIST', kind: 'forward' },
        { id: 'e5', source: 'n_conf', sourcePort: 'calibrated_uq', target: 'n_router', targetPort: 'posterior', signalType: 'UNCERTAINTY_DIST', kind: 'forward' },
        { id: 'e6', source: 'n_acq', sourcePort: 'selected_batch', target: 'n_router', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e7', source: 'n_router', sourcePort: 'tier1_batch', target: 'n_vina', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e8', source: 'n_pocket', sourcePort: 'pocket', target: 'n_vina', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e9', source: 'n_vina', sourcePort: 'scores', target: 'n_ledger', targetPort: 'scores', signalType: 'ORACLE_SCORES', kind: 'forward' },
        { id: 'e10', source: 'n_conf', sourcePort: 'calibrated_uq', target: 'n_ledger', targetPort: 'uncertainty', signalType: 'UNCERTAINTY_DIST', kind: 'forward' },
        { id: 'e11', source: 'n_router', sourcePort: 'routing_report', target: 'n_ledger', targetPort: 'reports', signalType: 'PROVENANCE_REPORT', kind: 'forward' },
        {
          id: 'e_fb_1',
          source: 'n_vina',
          sourcePort: 'scores',
          target: 'n_gp',
          targetPort: 'observations',
          signalType: 'ORACLE_SCORES',
          kind: 'feedback',
          label: '↻ DMTA Round t+1 Update',
        },
      ],
    },
  },
  {
    id: 'paralog_selectivity_mmp',
    shortLabel: 'Selectivity & MMP',
    name: 'Paralog Selectivity & Avoid-ome MMP Rescue',
    tagline:
      'Multi-conformer DRD1 vs. DRD2 contrastive CFG steering + 1-step makeable reaction chemistry + Roche-style Avoid-ome 1-cut MMP rescue + OpenFE RBFE.',
    graph: {
      schemaVersion: SCHEMA_VERSION,
      id: 'preset_paralog_selectivity_mmp',
      name: 'Paralog Selectivity & Avoid-ome MMP Rescue',
      description:
        'Parallel contrastive DRD1/DRD2 pocket steering and 1-step makeable amide enumeration feeding Avoid-ome 1-cut MMP bioisostere rescues, 12-endpoint Avoid-ome gating, and charge-matched OpenFE RBFE export.',
      metadata: {
        presetId: 'paralog_selectivity_mmp',
        author: 'GenEuclidean v5.1',
        tags: ['Selectivity', 'CFG', 'Makeable', 'MMP', 'OpenFE'],
      },
      execution: {
        mode: 'browser_sim',
        seed: 42,
        dmtaRounds: 1,
        targetContext: 'DRD1',
        budget: 32,
      },
      nodes: [
        // Row 1: Multi-Conformer -> PaiNN-SE(3) -> Ensemble CFG -> 12-Ep Avoid-ome
        {
          id: 'n_ens',
          blockType: 'multiconformer_ensemble',
          position: { x: 32, y: 40 },
          params: { on_target_id: 'DRD1', n_conformers: 2, off_target_id: 'DRD2', weighting: 'uniform' },
        },
        {
          id: 'n_painn',
          blockType: 'painn_se3',
          position: { x: 324, y: 40 },
          params: {
            num_layers: 4,
            hidden_dim: 192,
            vector_dim: 64,
            out_dim: 256,
            num_rbf: 24,
            cutoff: 5.5,
            max_neighbors: 24,
            use_se3_chirality: true,
          },
        },
        {
          id: 'n_cfg',
          blockType: 'pocket_ensemble_cfg',
          position: { x: 616, y: 40 },
          params: {
            backbone_id: 'datamol-io/safe-gpt',
            w_on: 1.6,
            w_off: 0.65,
            num_samples: 32,
            compute_pmi_3d: true,
          },
        },
        {
          id: 'n_admet',
          blockType: 'cns_admet_suite',
          position: { x: 908, y: 40 },
          params: {
            property_profile: 'cns_mpo_v2',
            admet_profile: 'v4_12ep',
            min_mpo_v2: 4.5,
            max_herg_prob: 0.25,
            require_triage_pass: true,
          },
        },
        // Row 2: Reagent Library -> 16-Rxn Synthesis -> Avoid-ome MMP -> OpenFE RBFE
        {
          id: 'n_reag',
          blockType: 'reagent_library',
          position: { x: 32, y: 312 },
          params: { core_smiles: 'O=C(O)c1ccc2ccccc2n1', reagent_pack: 'cns_amines', max_reagents: 32 },
        },
        {
          id: 'n_rxn',
          blockType: 'reaction_enumerator',
          position: { x: 324, y: 312 },
          params: {
            mode: 'enumerate_and_retro',
            reaction_family: 'amide_coupling_acid_amine',
            max_products: 64,
            triage_filter: true,
          },
        },
        {
          id: 'n_mmp',
          blockType: 'leadopt_mmp_sar',
          position: { x: 616, y: 312 },
          params: {
            mode: 'mmp_and_cliffs',
            property_profile: 'cns_mpo_v2',
            max_proposals: 20,
            cliff_threshold: 1.0,
            triage_filter: true,
          },
        },
        {
          id: 'n_openfe',
          blockType: 'openfe_rbfe_exporter',
          position: { x: 908, y: 312 },
          params: {
            require_same_charge: true,
            filter_ring_strain: true,
            max_ring_strain_kcal: 3.0,
            add_cycle_closures: true,
            out_dir: 'results/openfe_drd1_selectivity',
          },
        },
      ],
      edges: [
        { id: 'e1', source: 'n_ens', sourcePort: 'on_pockets', target: 'n_painn', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e2', source: 'n_painn', sourcePort: 'pocket_graph', target: 'n_cfg', targetPort: 'pocket_graph', signalType: 'SE3_GRAPH', kind: 'forward' },
        { id: 'e3', source: 'n_ens', sourcePort: 'off_pocket', target: 'n_cfg', targetPort: 'off_pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e4', source: 'n_reag', sourcePort: 'reagents', target: 'n_rxn', targetPort: 'reagents', signalType: 'REAGENTS', kind: 'forward' },
        { id: 'e5', source: 'n_rxn', sourcePort: 'makeable_mols', target: 'n_mmp', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e6', source: 'n_cfg', sourcePort: 'molecules', target: 'n_admet', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e7', source: 'n_mmp', sourcePort: 'rescued_mols', target: 'n_admet', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e8', source: 'n_admet', sourcePort: 'safe_mols', target: 'n_openfe', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e9', source: 'n_ens', sourcePort: 'on_pockets', target: 'n_openfe', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
      ],
    },
  },
  {
    id: 'boltz2_audit_conformal_dmpk',
    shortLabel: 'Boltz-2 7-Axis',
    name: 'Boltz-2 / AF3 7-Axis Audit & Conformal DMPK',
    tagline:
      'Zero-PyTorch QA circuit auditing a co-folded ACHE complex across Cremer–Pople ring/torsion strain, 7-Axis PoseRealityAuditor, 90% Conformal DMPK & MCP tools.',
    graph: {
      schemaVersion: SCHEMA_VERSION,
      id: 'preset_boltz2_audit_conformal_dmpk',
      name: 'Boltz-2 / AF3 7-Axis Audit & Conformal DMPK',
      description:
        'Zero-PyTorch industrial pose quality-assurance circuit auditing a Boltz-2 / AF3 co-folded ACHE complex across ring/torsion strain, 7-Axis PoseRealityAuditor, 90% Split-Conformal DMPK, 3D Studio, and SHA-256 MCP tools.',
      metadata: {
        presetId: 'boltz2_audit_conformal_dmpk',
        author: 'GenEuclidean v5.1',
        tags: ['Boltz-2', '7-Axis', 'Conformal DMPK', 'MCP'],
      },
      execution: {
        mode: 'browser_sim',
        seed: 42,
        dmtaRounds: 1,
        targetContext: 'ACHE',
        budget: 16,
      },
      nodes: [
        {
          id: 'n_boltz',
          blockType: 'boltz2_cofold',
          position: { x: 24, y: 168 },
          params: {
            structure_source: 'ACHE_donepezil_cofold.cif',
            ligand_resname: 'LIG',
            reference_smiles: 'COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2',
            pocket_cutoff: 10.0,
          },
        },
        {
          id: 'n_strain',
          blockType: 'ring_torsion_strain',
          position: { x: 304, y: 36 },
          params: {
            ring_strain_threshold_kcal: 3.0,
            ring_n_ref_confs: 24,
            torsion_barrier_threshold_kcal: 2.5,
            torsion_max_red: 0,
            torsion_backend: 'smarts_mmff94s',
          },
        },
        {
          id: 'n_prot',
          blockType: 'protonation_ph74',
          position: { x: 304, y: 304 },
          params: {
            audit_tautomers: true,
            max_tautomers: 16,
            tautomer_gap_threshold: 2.0,
            strict_input_charge: false,
          },
        },
        {
          id: 'n_audit',
          blockType: 'pose_reality_auditor',
          position: { x: 584, y: 36 },
          params: {
            property_profile: 'cns_mpo_v2',
            ring_strain_threshold_kcal: 3.0,
            torsion_barrier_threshold_kcal: 2.5,
            torsion_max_red: 0,
            strict_input_charge: false,
          },
        },
        {
          id: 'n_dmpk',
          blockType: 'conformal_dmpk',
          position: { x: 584, y: 304 },
          params: { endpoint: 'OpenADMET_LogD', alpha: 0.1, n_estimators: 80 },
        },
        {
          id: 'n_studio3d',
          blockType: 'studio_3d_inspector',
          position: { x: 864, y: 36 },
          params: { theme: 'dark', show_native_ghost: true, highlight_anchors: true },
        },
        {
          id: 'n_mcp',
          blockType: 'mcp_tool_export',
          position: { x: 864, y: 304 },
          params: { tool_name: 'audit_complex', transport: 'python_direct' },
        },
      ],
      edges: [
        { id: 'e1', source: 'n_boltz', sourcePort: 'pose', target: 'n_strain', targetPort: 'poses', signalType: 'POSE_3D', kind: 'forward' },
        { id: 'e2', source: 'n_strain', sourcePort: 'clean_poses', target: 'n_audit', targetPort: 'poses', signalType: 'POSE_3D', kind: 'forward' },
        { id: 'e3', source: 'n_boltz', sourcePort: 'pocket', target: 'n_audit', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e4', source: 'n_boltz', sourcePort: 'ligand_mols', target: 'n_prot', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e5', source: 'n_prot', sourcePort: 'protonated_mols', target: 'n_dmpk', targetPort: 'molecules', signalType: 'MOL_BATCH', kind: 'forward' },
        { id: 'e6', source: 'n_boltz', sourcePort: 'pocket', target: 'n_studio3d', targetPort: 'pocket', signalType: 'POCKET_3D', kind: 'forward' },
        { id: 'e7', source: 'n_audit', sourcePort: 'verified_poses', target: 'n_studio3d', targetPort: 'poses', signalType: 'POSE_3D', kind: 'forward' },
        { id: 'e8', source: 'n_audit', sourcePort: 'audit_report', target: 'n_studio3d', targetPort: 'audit_report', signalType: 'PROVENANCE_REPORT', kind: 'forward' },
        { id: 'e9', source: 'n_audit', sourcePort: 'audit_report', target: 'n_mcp', targetPort: 'audit_report', signalType: 'PROVENANCE_REPORT', kind: 'forward' },
        { id: 'e10', source: 'n_dmpk', sourcePort: 'dmpk_intervals', target: 'n_mcp', targetPort: 'uncertainty', signalType: 'UNCERTAINTY_DIST', kind: 'forward' },
      ],
    },
  },
  {
    id: 'blank_breadboard',
    shortLabel: 'Blank',
    name: 'Blank Breadboard',
    tagline:
      'Start from an empty canvas — drag or click any of the 36 GenEuclidean blocks from the dock to compose your own discovery circuit.',
    graph: {
      schemaVersion: SCHEMA_VERSION,
      id: 'preset_blank_breadboard',
      name: 'Blank Breadboard',
      description: 'Custom GenEuclidean drug-discovery circuit.',
      metadata: {
        presetId: 'blank_breadboard',
        author: 'Custom',
        tags: ['Custom'],
      },
      execution: {
        mode: 'browser_sim',
        seed: 42,
        dmtaRounds: 1,
        targetContext: 'DRD1',
        budget: 16,
      },
      nodes: [],
      edges: [],
    },
  },
];

export function cloneCircuitGraph(graph: CircuitGraphV1): CircuitGraphV1 {
  return JSON.parse(JSON.stringify(graph)) as CircuitGraphV1;
}

// Ergonomic aliases & theme-aware color helpers
export const BLOCK_MAP = BLOCK_CATALOG_MAP;
export const SIGNAL_META = SIGNAL_TYPE_META;
export const getNodeParamSummary = formatNodeParamSummary;

export function getSignalColor(signalType: SignalType, theme: 'dark' | 'light'): string {
  const meta = SIGNAL_TYPE_META[signalType];
  if (!meta) return theme === 'dark' ? '#38BDF8' : '#0284C7';
  return theme === 'dark' ? meta.darkHex : meta.lightHex;
}

export function getCategoryColor(category: BlockCategoryId, theme: 'dark' | 'light'): string {
  const meta = CATEGORY_META_MAP[category];
  if (!meta) return theme === 'dark' ? '#38BDF8' : '#0284C7';
  return theme === 'dark' ? meta.darkHex : meta.lightHex;
}

export function createNodeFromBlock(
  blockType: string,
  position: { x: number; y: number },
  existingIds: Set<string>,
): CircuitNode | null {
  const def = BLOCK_CATALOG_MAP[blockType];
  if (!def) return null;
  const base = `n_${blockType.split('_')[0]}`;
  let idx = 1;
  let id = `${base}_${idx}`;
  while (existingIds.has(id)) {
    idx += 1;
    id = `${base}_${idx}`;
  }
  return {
    id,
    blockType,
    position: { x: Math.round(position.x), y: Math.round(position.y) },
    params: createDefaultParams(blockType),
  };
}
