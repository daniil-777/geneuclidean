export type CircuitSchemaVersion = 'geneuclidean.circuit.v1';

export type SignalType =
  | 'POCKET_3D'
  | 'SE3_GRAPH'
  | 'LATENT_VEC'
  | 'MOL_BATCH'
  | 'POSE_3D'
  | 'ORACLE_SCORES'
  | 'UNCERTAINTY_DIST'
  | 'REAGENTS'
  | 'RBFE_GRAPH'
  | 'PROVENANCE_REPORT';

export type BlockCategoryId =
  | 'inputs'
  | 'encoders'
  | 'generators'
  | 'screening'
  | 'active_learning'
  | 'physics_audit'
  | 'leadopt_admet'
  | 'outputs';

export type EdgeKind = 'forward' | 'feedback';
export type ExecutionEngineMode = 'browser_sim' | 'local_python';

export interface SignalTypeMeta {
  type: SignalType;
  label: string;
  shortLabel: string;
  darkHex: string;
  lightHex: string;
  pythonContract: string;
}

export interface BlockCategoryMeta {
  id: BlockCategoryId;
  label: string;
  shortLabel: string;
  darkHex: string;
  lightHex: string;
  description: string;
}

export interface PortSpec {
  id: string;
  label: string;
  signalType: SignalType;
  direction: 'input' | 'output';
  required?: boolean;
  multi?: boolean;
  acceptsFeedback?: boolean;
  shapeHint?: string;
}

export interface ParamSpec {
  key: string;
  label: string;
  type: 'number' | 'select' | 'boolean' | 'string';
  default: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: (string | number)[];
  description?: string;
}

export interface BlockDefinition {
  blockType: string;
  name: string;
  shortName: string;
  category: BlockCategoryId;
  pythonSymbol: string;
  executionTier: 'zero_torch' | 'optional_torch' | 'v51_al';
  badge: string;
  summary: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  params: ParamSpec[];
}

export interface CircuitNode {
  id: string;
  blockType: string;
  label?: string;
  position: { x: number; y: number };
  params: Record<string, number | string | boolean>;
  collapsed?: boolean;
  disabled?: boolean;
}

export interface CircuitEdge {
  id: string;
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
  signalType: SignalType;
  kind: EdgeKind;
  label?: string | null;
}

export interface CircuitGraphV1 {
  schemaVersion: CircuitSchemaVersion;
  id: string;
  name: string;
  description: string;
  metadata?: {
    presetId?: string;
    author?: string;
    createdAt?: string;
    tags?: string[];
  };
  execution: {
    mode: ExecutionEngineMode;
    seed: number;
    dmtaRounds: number;
    targetContext: string;
    budget: number;
  };
  viewport?: { x: number; y: number; zoom: number };
  nodes: CircuitNode[];
  edges: CircuitEdge[];
}

export interface CircuitPreset {
  id: string;
  shortLabel: string;
  name: string;
  tagline: string;
  badge?: string;
  graph: CircuitGraphV1;
}

export type ValidationDiagnosticCode =
  | 'E_UNKNOWN_BLOCK'
  | 'E_UNKNOWN_PORT'
  | 'E_SELF_LOOP'
  | 'E_PORT_TYPE_MISMATCH'
  | 'E_DUPLICATE_EDGE'
  | 'E_MULTI_INPUT_DISALLOWED'
  | 'E_FEEDBACK_NOT_ALLOWED'
  | 'E_COMBINATIONAL_CYCLE'
  | 'E_MISSING_REQUIRED_INPUT'
  | 'E_PARAM_OUT_OF_RANGE'
  | 'W_DISCONNECTED_NODE'
  | 'I_EMPTY_CIRCUIT';

export interface ValidationDiagnostic {
  severity: 'error' | 'warning' | 'info';
  code: ValidationDiagnosticCode;
  message: string;
  remediation?: string;
  nodeId?: string;
  edgeId?: string;
  portId?: string;
}

export interface ValidationResult {
  valid: boolean;
  topologicalOrder: string[];
  feedbackEdges: string[];
  hasFeedbackLoop: boolean;
  diagnostics: ValidationDiagnostic[];
}

export interface CodegenResult {
  valid: boolean;
  pythonCode: string;
  imports: string[];
  astValid: boolean;
  provenanceSha256: string;
  diagnostics: ValidationDiagnostic[];
}

export interface ExecutionStepTelemetry {
  round: number;
  nodeId: string;
  blockType: string;
  status: 'ok' | 'simulated_fallback' | 'skipped' | 'error';
  durationMs: number;
  executionBackend: 'real_python' | 'deterministic_surrogate';
  inputShapes: Record<string, string>;
  outputShapes: Record<string, string>;
  metrics: Record<string, number | string | boolean | null>;
  summary: string;
}

export interface CircuitMoleculeOutput {
  id: string;
  name: string;
  smiles: string;
  target: string;
  sourceNodeId: string;
  metrics: {
    vina_dg?: number | null;
    cns_mpo_v2?: number | null;
    herg_prob?: number | null;
    bbb_prob?: number | null;
    mdr1_prob?: number | null;
    reality_score?: number | null;
    posebusters_passed?: boolean | null;
    pred_pkd?: number | null;
    conformal_lo?: number | null;
    conformal_hi?: number | null;
    reaction_or_transform?: string | null;
  };
}

export interface DmtaRoundPoint {
  round: number;
  nObserved: number;
  bestValue: number;
  meanAcquired: number;
  topHitRecall: number;
  conformalCoverage: number;
  eiMax: number;
  topKJaccard: number;
}

export interface RunCircuitResponse {
  status: 'completed' | 'failed' | 'validation_error';
  engineMode: ExecutionEngineMode;
  executionId: string;
  provenanceSha256: string;
  totalDurationMs: number;
  roundsCompleted: number;
  diagnostics: ValidationDiagnostic[];
  steps: ExecutionStepTelemetry[];
  molecules: CircuitMoleculeOutput[];
  dmtaTrace: DmtaRoundPoint[];
  provenanceLedger: {
    schema: 'geneuclidean_dmta_ledger_v1';
    graphSha256: string;
    resultSha256: string;
    timestampUtc: string;
    fdaEmaPrinciples: string;
  };
  generatedCode: string;
}

export interface LocalEngineHealth {
  reachable: boolean;
  checking: boolean;
  version?: string;
  python?: string;
  rdkit?: boolean;
  torch?: boolean;
  activeLearningV51?: boolean;
  catalogCount?: number;
  error?: string;
}

export type NodeRuntimeStatus = 'idle' | 'running' | 'ok' | 'simulated_fallback' | 'error';

export type RightDrawerTab = 'inspector' | 'python' | 'telemetry';

export interface WiringDraft {
  sourceNodeId: string;
  sourcePortId: string;
  signalType: SignalType;
  startX: number;
  startY: number;
  cursorX: number;
  cursorY: number;
  hoveredTarget: {
    nodeId: string;
    portId: string;
    compatible: boolean;
    willBeFeedback: boolean;
    hint?: string;
  } | null;
}

export type ActiveWiringDraft = WiringDraft;
