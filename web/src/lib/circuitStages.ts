// A read-only view of the executable graph. Stage membership never changes saved nodes or edges.
import { BLOCK_CATALOG_MAP } from './circuitCatalog.ts';
import type { BlockCategoryId, CircuitEdge, CircuitGraphV1, CircuitNode } from '../types/circuit';

export type StageId = 'input' | 'encoder' | 'decoder' | 'smiles' | 'screening' | 'learning'
  | 'physics' | 'optimization' | 'output' | 'other';

export interface CircuitStage {
  id: StageId;
  label: string;
  description: string;
  nodes: CircuitNode[];
  internal: CircuitEdge[];
  incoming: CircuitEdge[];
  outgoing: CircuitEdge[];
  products: { nodeId: string; portId: string }[];
}

export interface StageLink {
  source: StageId;
  target: StageId;
  kind: 'forward' | 'feedback' | 'product';
  edges: CircuitEdge[];
}

export interface StageProjection {
  primary: CircuitStage[];
  additional: CircuitStage[];
  stages: CircuitStage[];
  links: StageLink[];
  stageByNode: ReadonlyMap<string, StageId>;
  unresolved: CircuitEdge[];
}

const stageText: Record<StageId, readonly [string, string]> = {
  input: ['Inputs', 'Structures and molecules supplied to the circuit.'],
  encoder: ['Encoder', 'Turn molecular or pocket geometry into a representation for later blocks.'],
  decoder: ['Decoder', 'Use the conditioning inputs to propose or refine molecular sequences.'],
  smiles: ['SMILES', 'Candidate molecules represented as text at the decoder output.'],
  screening: ['Screening', 'Search, rank and match candidate molecules.'],
  learning: ['Active learning', 'Model observations, estimate uncertainty and select experiments.'],
  physics: ['Structure & physics', 'Model 3D structures and evaluate candidate poses.'],
  optimization: ['Lead optimization', 'Propose changes or evaluate molecular properties.'],
  output: ['Outputs', 'Inspect and export the results of this circuit.'],
  other: ['Other blocks', 'Imported blocks without a recognized catalog category.'],
};
const categoryStage: Record<BlockCategoryId, StageId> = {
  inputs: 'input', encoders: 'encoder', generators: 'decoder', screening: 'screening',
  active_learning: 'learning', physics_audit: 'physics', leadopt_admet: 'optimization', outputs: 'output',
};

function stageOf(node: CircuitNode): StageId {
  if (node.blockType === 'nlp_perceiver_bridge') return 'encoder'; // Outputs pocket conditioning tokens.
  if (node.blockType === 'selectivity_mtl_aligner') return 'optimization'; // Outputs a tuned vector.
  if (!Object.hasOwn(BLOCK_CATALOG_MAP, node.blockType)) return 'other';
  return categoryStage[BLOCK_CATALOG_MAP[node.blockType].category];
}

// Feedback runs in a subsequent round. Sorting the forward dependencies only also works
// for disconnected graphs. Keep malformed forward cycles visible in their original order.
function forwardOrder<K extends string>(ids: K[], pairs: Iterable<readonly [K, K]>): K[] {
  const incoming = new Map(ids.map(id => [id, 0]));
  const next = new Map<K, K[]>();
  for (const [source, target] of pairs) {
    if (source === target || !incoming.has(source) || !incoming.has(target)) continue;
    const targets = next.get(source) ?? [];
    targets.push(target);
    next.set(source, targets);
    incoming.set(target, incoming.get(target)! + 1);
  }
  const ordered = ids.filter(id => incoming.get(id) === 0);
  for (let i = 0; i < ordered.length; i++) {
    for (const target of next.get(ordered[i]) ?? []) {
      const count = incoming.get(target)! - 1;
      incoming.set(target, count);
      if (count === 0) ordered.push(target);
    }
  }
  if (ordered.length !== ids.length) {
    const seen = new Set(ordered);
    ordered.push(...ids.filter(id => !seen.has(id)));
  }
  return ordered;
}

export function projectCircuitStages(graph: CircuitGraphV1): StageProjection {
  const groups = new Map<StageId, CircuitStage>();
  function stage(id: StageId): CircuitStage {
    const existing = groups.get(id);
    if (existing) return existing;
    const [label, description] = stageText[id];
    const created: CircuitStage = { id, label, description, nodes: [], internal: [], incoming: [], outgoing: [], products: [] };
    groups.set(id, created);
    return created;
  }

  const nodes = new Map(graph.nodes.map(node => [node.id, node]));
  const stageByNode = new Map<string, StageId>();
  const forwardEdges = graph.edges.filter(edge => edge.kind === 'forward');
  for (const id of forwardOrder([...nodes.keys()], forwardEdges.map(edge => [edge.source, edge.target]))) {
    const node = nodes.get(id)!;
    const group = stage(stageOf(node));
    group.nodes.push(node);
    stageByNode.set(id, group.id);
  }
  const input = groups.get('input');
  if (input?.nodes.every(node => ['protein_pocket', 'multiconformer_ensemble'].includes(node.blockType))) {
    input.label = 'Pocket';
    input.description = 'The 3D binding-site atoms and coordinates supplied to the circuit.';
  }

  const aggregate = new Map<string, StageLink>();
  const unresolved: CircuitEdge[] = [];
  // A decoder port that is consumed only by another decoder is an intermediate.
  // Keep ports that also feed an external stage, as well as unconnected output ports.
  const downstream = new Map<string, Map<string, Set<StageId>>>();
  for (const edge of graph.edges) {
    const source = stageByNode.get(edge.source);
    const target = stageByNode.get(edge.target);
    if (!source || !target) unresolved.push(edge);
    if (source && source === target) {
      stage(source).internal.push(edge);
    } else {
      if (source) stage(source).outgoing.push(edge);
      if (target) stage(target).incoming.push(edge);
      if (source && target) {
        const key = source + '/' + target + '/' + edge.kind;
        const link = aggregate.get(key) ?? { source, target, kind: edge.kind, edges: [] };
        link.edges.push(edge);
        aggregate.set(key, link);
      }
    }
    if (source === 'decoder' && target && !nodes.get(edge.target)?.disabled &&
        edge.kind === 'forward' && edge.signalType === 'MOL_BATCH') {
      const ports = downstream.get(edge.source) ?? new Map<string, Set<StageId>>();
      const targets = ports.get(edge.sourcePort) ?? new Set<StageId>();
      targets.add(target);
      ports.set(edge.sourcePort, targets);
      downstream.set(edge.source, ports);
    }
  }

  for (const node of groups.get('decoder')?.nodes ?? []) {
    if (node.disabled) continue;
    for (const port of BLOCK_CATALOG_MAP[node.blockType].outputs) {
      if (port.signalType !== 'MOL_BATCH') continue;
      const targets = downstream.get(node.id)?.get(port.id);
      if (targets?.size === 1 && targets.has('decoder')) continue;
      stage('smiles').products.push({ nodeId: node.id, portId: port.id });
    }
  }
  const links = [...aggregate.values()];
  if (groups.has('smiles')) {
    links.push({ source: 'decoder', target: 'smiles', kind: 'product', edges: [] });
  }
  const groupOrder = forwardOrder([...groups.keys()], links
    .filter(link => link.kind !== 'feedback').map(link => [link.source, link.target]));
  const stages = groupOrder.map(id => groups.get(id)!);
  const primary = groups.has('smiles')
    ? (['input', 'encoder', 'decoder', 'smiles'] as StageId[]).flatMap(id => groups.get(id) ?? [])
    : stages.slice(0, 4);
  return { primary, additional: stages.filter(s => !primary.includes(s)), stages, links, stageByNode, unresolved };
}

export interface ImplementationStep {
  label: string;
  description: string;
  source: string;
}

const implementation: Record<string, ImplementationStep[]> = {
  painn_se3: [
    { label: 'Build a neighbor graph', description: 'Connect nearby pocket atoms using the radius cutoff and neighbor limit.', source: 'src/geneuclidean/models/encoders/painn.py#L266' },
    { label: 'Embed pocket atoms', description: 'Map atom features into scalar channels and expand interatomic distances into radial features.', source: 'src/geneuclidean/models/encoders/painn.py#L274' },
    { label: 'Exchange geometric messages', description: 'Update scalar and vector channels using neighbor features and edge directions.', source: 'src/geneuclidean/models/encoders/painn.py#L290' },
    { label: 'Read out the pocket', description: 'Return per-atom features and a masked attention pool over the pocket.', source: 'src/geneuclidean/models/encoders/painn.py#L296' },
  ],
  safe_gpt: [
    { label: 'Build the pocket prefix', description: 'Cross-attention resamples the encoded pocket atoms into conditioning tokens for the language model.', source: 'src/geneuclidean/models/llm_bridge.py#L892' },
    { label: 'Sample SAFE tokens', description: 'The conditioned language model selects the next token until a stop token or the length limit.', source: 'src/geneuclidean/sampling/llm_sampler.py#L247' },
    { label: 'Decode to SMILES', description: 'Decode the generated SAFE fragment sequence into a molecular string.', source: 'src/geneuclidean/models/llm_bridge.py#L449' },
  ],
};

export function getBlockImplementation(blockType: string): readonly ImplementationStep[] {
  return Object.hasOwn(implementation, blockType) ? implementation[blockType] : [];
}
