import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { CIRCUIT_PRESETS } from '../src/lib/circuitCatalog.ts';
import { getBlockImplementation, projectCircuitStages } from '../src/lib/circuitStages.ts';

function preset(id) {
  return structuredClone(CIRCUIT_PRESETS.find(p => p.id === id).graph);
}

for (const { id, graph } of CIRCUIT_PRESETS) {
  test(id + ' projects all blocks and preserves every actual wire, port and direction', () => {
    const original = structuredClone(graph);
    const view = projectCircuitStages(graph);
    assert.deepEqual(graph, original);
    assert.ok(view.primary.length <= 4);
    assert.deepEqual(new Set([...view.primary, ...view.additional]), new Set(view.stages));
    assert.equal(view.primary.length + view.additional.length, view.stages.length);
    assert.deepEqual(view.unresolved, []);
    const nodes = view.stages.flatMap(s => s.nodes);
    assert.equal(nodes.length, graph.nodes.length);
    assert.equal(new Set(nodes).size, nodes.length);
    assert.deepEqual(new Set(nodes), new Set(graph.nodes));
    const crossing = [];
    for (const e of graph.edges) {
      const source = view.stages.find(s => s.id === view.stageByNode.get(e.source));
      const target = view.stages.find(s => s.id === view.stageByNode.get(e.target));
      if (source === target) {
        assert.ok(source.internal.includes(e));
      } else {
        assert.ok(source.outgoing.includes(e));
        assert.ok(target.incoming.includes(e));
        crossing.push(e);
      }
    }
    const linkEdges = view.links.flatMap(l => l.edges);
    assert.equal(linkEdges.length, crossing.length);
    assert.deepEqual(new Set(linkEdges), new Set(crossing));
    for (const link of view.links) {
      if (link.kind === 'product') {
        assert.deepEqual([link.source, link.target, link.edges], ['decoder', 'smiles', []]);
      }
      for (const edge of link.edges) {
        assert.deepEqual([link.source, link.target, link.kind],
          [view.stageByNode.get(edge.source), view.stageByNode.get(edge.target), edge.kind]);
      }
    }
  });
}

test('champion stages expose real subblocks and only the outward-facing decoded molecule batch', () => {
  const graph = preset('tourette_scz_champion');
  // Visual or import ordering must not put consumers ahead of their producers.
  graph.nodes.reverse();
  const { primary, links } = projectCircuitStages(graph);
  assert.deepEqual(primary.map(s => s.label), ['Pocket', 'Encoder', 'Decoder', 'SMILES']);
  assert.deepEqual(primary[1].nodes.map(n => n.blockType), ['cartformer_se3', 'nlp_perceiver_bridge']);
  assert.deepEqual(primary[2].nodes.map(n => n.blockType), ['nlp_rope_gqa_lm', 'pocket_ensemble_cfg', 'nlp_lora_rl_aligner']);
  assert.deepEqual(primary[3].products, [{ nodeId: 'n_rl', portId: 'aligned_mols' }]);
  assert.deepEqual(primary[3].nodes, []);
  assert.equal(links.filter(l => l.kind === 'product').length, 1);

  // Branching an intermediate sequence batch out of the decoder makes it an
  // additional externally visible product and keeps both port names.
  graph.edges.push({ ...graph.edges.find(e => e.id === 'e9'), id: 'custom-edge', source: 'n_lm', sourcePort: 'molecules' });
  assert.deepEqual(projectCircuitStages(graph).primary.at(-1).products,
    [{ nodeId: 'n_lm', portId: 'molecules' }, { nodeId: 'n_rl', portId: 'aligned_mols' }]);
});

test('a molecular library input or folded pose does not imply SMILES generation', () => {
  for (const id of ['al_dmta_closed_loop', 'nanofold3_cofold_audit', 'boltz2_audit_conformal_dmpk']) {
    const { stages, links } = projectCircuitStages(preset(id));
    assert.equal(stages.some(s => s.id === 'smiles' || s.id === 'decoder'), false);
    assert.equal(links.some(l => l.kind === 'product'), false);
  }
  const closedLoop = projectCircuitStages(preset('al_dmta_closed_loop'));
  assert.ok(closedLoop.links.some(l => l.kind === 'feedback') || closedLoop.stages.some(s => s.internal.some(e => e.kind === 'feedback')));
});

test('only enabled consumers determine which decoder outputs are exposed', () => {
  const graph = preset('tourette_scz_champion');
  const aligner = graph.nodes.find(n => n.id === 'n_rl');
  aligner.disabled = true;
  const view = projectCircuitStages(graph);
  const decoder = view.stages.find(s => s.id === 'decoder');
  assert.deepEqual(view.stages.find(s => s.id === 'smiles').products, [{ nodeId: 'n_cfg', portId: 'molecules' }]);
  assert.ok(decoder.internal.includes(graph.edges.find(e => e.id === 'e7')));

  // A disabled external consumer cannot turn a live intermediate into a separate product.
  aligner.disabled = false;
  graph.nodes.push({ ...graph.nodes.find(n => n.id === 'n_vina'), id: 'inactive_vina', disabled: true });
  graph.edges.push({ ...graph.edges.find(e => e.id === 'e9'), id: 'also-vina',
    source: 'n_cfg', sourcePort: 'molecules', target: 'inactive_vina' });
  assert.deepEqual(projectCircuitStages(graph).stages.find(s => s.id === 'smiles').products,
    [{ nodeId: 'n_rl', portId: 'aligned_mols' }]);
});

test('a blank or disconnected graph does not acquire a workflow from its preset ID', () => {
  const graph = preset('se3_denovo_cns');
  graph.edges = [];
  let projection = projectCircuitStages(graph);
  assert.deepEqual(projection.links.map(l => l.kind), ['product']);
  graph.nodes.find(n => n.blockType === 'safe_gpt').disabled = true;
  projection = projectCircuitStages(graph);
  assert.equal(projection.links.length, 0);
  assert.equal(projection.stages.some(s => s.id === 'smiles'), false);
  graph.nodes = [];
  assert.deepEqual(projectCircuitStages(graph).primary, []);
});

test('unrecognized imported blocks and dangling connections stay inspectable', () => {
  const graph = preset('blank_breadboard');
  graph.nodes = [{ id: 'custom', blockType: '__proto__', params: {}, position: { x: 15, y: 15 } }];
  graph.edges = [{ id: 'dangling', source: 'gone', sourcePort: 'out', target: 'custom', targetPort: 'in', signalType: 'LATENT_VEC', kind: 'forward' }];
  const projection = projectCircuitStages(graph);
  assert.equal(projection.primary[0].id, 'other');
  assert.deepEqual(projection.unresolved, graph.edges);
  assert.deepEqual(projection.primary[0].incoming, graph.edges);
  assert.deepEqual(projection.links, []);
});

test('singleton implementation steps are documented Python internals with source locations', () => {
  for (const key of ['painn_se3', 'safe_gpt']) {
    const steps = getBlockImplementation(key);
    assert.ok(steps.length >= 3);
    for (const step of steps) {
      const [path, line] = step.source.split('#L');
      const text = readFileSync(new URL('../../' + path, import.meta.url), 'utf8');
      assert.ok(text.split('\n')[Number(line) - 1]?.trim(), step.source);
    }
  }
  assert.deepEqual(getBlockImplementation('nlp_lora_rl_aligner'), []);
  assert.deepEqual(getBlockImplementation('__proto__'), []);
});
