import assert from 'node:assert/strict';
import test from 'node:test';
import { CIRCUIT_PRESETS } from '../src/lib/circuitCatalog.ts';
import { projectCircuitStages } from '../src/lib/circuitStages.ts';
import { layoutStageDiagram } from '../src/lib/stageDiagram.ts';

const within = (n, a, b) => n >= Math.min(a, b) && n <= Math.max(a, b);
const overlap = (a, b, c, d) => Math.min(Math.max(a, b), Math.max(c, d)) - Math.max(Math.min(a, b), Math.min(c, d));
const vertices = points => points.slice(1).map((p, i) => [points[i], p]);

function paintedSegments(path) {
  const segments = [];
  let prev = null;
  for (const [, cmd, x, y] of path.matchAll(/([ML])(-?[\d.]+) (-?[\d.]+)/g)) {
    const next = { x: Number(x), y: Number(y) };
    if (cmd === 'L') segments.push([prev, next]);
    prev = next;
  }
  return segments;
}

// Disallow shared painted pixels between centerlines, apart from marker/decorative stroke width.
// Test the actual geometry passed to SVG, including any gaps at rail crossings.
function touches([a, b], [c, d]) {
  const h1 = a.y === b.y, h2 = c.y === d.y;
  if (h1 && h2) return a.y === c.y && overlap(a.x, b.x, c.x, d.x) >= 0;
  if (!h1 && !h2) return a.x === c.x && overlap(a.y, b.y, c.y, d.y) >= 0;
  if (!h1) return touches([c, d], [a, b]);
  return within(c.x, a.x, b.x) && within(a.y, c.y, d.y);
}

function throughCard([a, b], card) {
  if (a.x === b.x) return a.x > card.x && a.x < card.x + card.width &&
    overlap(a.y, b.y, card.y, card.y + card.height) > 0;
  assert.equal(a.y, b.y, 'all routes must be orthogonal');
  return a.y > card.y && a.y < card.y + card.height && overlap(a.x, b.x, card.x, card.x + card.width) > 0;
}

function checkGeometry(diagram, nodes, edges) {
  assert.deepEqual(diagram.nodes.map(n => n.id), nodes.map(n => n.id));
  const byId = new Map(diagram.nodes.map(n => [n.id, n]));
  const internal = edges.filter(e => byId.has(e.source) && byId.has(e.target));
  assert.deepEqual(new Set(diagram.edges.map(e => e.id)), new Set(internal.map(e => e.id)));
  assert.equal(diagram.nodes.reduce((n, card) => n + card.ports.length, 0), internal.length * 2);
  assert.equal(diagram.edges.length, internal.length);
  const socketPositions = new Set();
  const allPaint = [];
  for (const e of diagram.edges) {
    const source = byId.get(e.source), target = byId.get(e.target);
    assert.ok(source && target);
    const src = source.ports.find(p => p.edgeId === e.id && p.direction === 'output');
    const dst = target.ports.find(p => p.edgeId === e.id && p.direction === 'input');
    assert.equal(src.portId, e.sourcePort);
    assert.equal(dst.portId, e.targetPort);
    for (const [card, port, vertex] of [[source, src, e.points[0]], [target, dst, e.points.at(-1)]]) {
      assert.deepEqual(vertex, { x: port.x, y: port.y });
      assert.ok(within(port.x, card.x, card.x + card.width) && within(port.y, card.y, card.y + card.height));
      const side = { left: port.x === card.x, top: port.y === card.y,
        right: port.x === card.x + card.width, bottom: port.y === card.y + card.height };
      assert.ok(side[port.side], 'socket lies on its declared side');
      const key = port.x + ',' + port.y;
      assert.ok(!socketPositions.has(key), 'each wire gets a distinct socket');
      socketPositions.add(key);
    }
    for (const p of e.points) {
      assert.ok(within(p.x, 0, diagram.width) && within(p.y, 0, diagram.height));
    }
    for (const segment of vertices(e.points)) {
      for (const card of diagram.nodes) assert.ok(!throughCard(segment, card), 'wire crosses ' + card.id);
    }
    const painted = paintedSegments(e.path);
    assert.ok(painted.length >= 1);
    assert.deepEqual(painted[0][0], e.points[0]);
    assert.deepEqual(painted.at(-1)[1], e.points.at(-1));
    for (const [wire, previous] of allPaint) {
      assert.ok(!previous.some(p => painted.some(s => touches(s, p))), wire + ' touches ' + e.id);
    }
    allPaint.push([e.id, painted]);
  }
}

const nodes = 'ABCDEFG'.split('').map(id => ({ id, blockType: 'test', params: {}, position: { x: 0, y: 0 } }));
const wire = (id, source, target, kind = 'forward') => ({ id, source, target, kind,
  sourcePort: 'out-' + id, targetPort: 'in-' + id, signalType: 'LATENT_VEC' });

for (const orientation of ['vertical', 'horizontal']) {
  test(orientation + ' diagrams fit every preset and preserve exact port-level connections', () => {
    for (const { graph } of CIRCUIT_PRESETS) {
      for (const stage of projectCircuitStages(graph).stages) {
        const diagram = layoutStageDiagram(stage.nodes, stage.internal, orientation);
        checkGeometry(diagram, stage.nodes, stage.internal);
        if (orientation === 'vertical') assert.ok(diagram.width <= 300, 'a preset exceeds compact width');
      }
    }
  });

  test(orientation + ' champion decoder separates the molecule and reward channels without crossings', () => {
    const champion = CIRCUIT_PRESETS.find(p => p.id === 'tourette_scz_champion');
    const decoder = projectCircuitStages(champion.graph).stages.find(s => s.id === 'decoder');
    const d = layoutStageDiagram(decoder.nodes, decoder.internal, orientation);
    assert.equal(d.edges.length, 3);
    for (const e of d.edges) {
      assert.equal(e.points.length, 2);
      assert.equal(e.path, e.hitPath);
    }
    assert.deepEqual(d.edges.filter(e => e.source === 'n_cfg').map(e => [e.sourcePort, e.targetPort]),
      [['molecules', 'molecules'], ['pmi_scores', 'rewards']]);
    if (orientation === 'horizontal') assert.ok(d.width < 700);
  });

  test(orientation + ' fans, parallel branches, feedback, self-loops and disconnected nodes stay distinct', () => {
    const edges = [wire('fan1', 'A', 'C'), wire('fan2', 'A', 'E'), wire('again', 'A', 'E'),
      wire('cross1', 'B', 'E'), wire('cross2', 'C', 'F'), wire('direct', 'D', 'E'),
      wire('reverse', 'F', 'A', 'feedback'), wire('reverse2', 'E', 'B', 'feedback'),
      wire('self', 'C', 'C', 'feedback'), wire('dangling', 'X', 'E'), wire('external', 'C', 'X')];
    const diagram = layoutStageDiagram(nodes, edges, orientation);
    checkGeometry(diagram, nodes, edges);
    assert.ok(diagram.edges.some(e => e.path !== e.hitPath), 'unavoidable crossings have a visible break');
    assert.deepEqual(nodes.map(n => n.position), nodes.map(() => ({ x: 0, y: 0 })));
    assert.deepEqual(layoutStageDiagram(nodes, edges.toReversed(), orientation), diagram);
    assert.deepEqual(diagram.nodes.find(n => n.id === 'G').ports, []);
  });

  test(orientation + ' dense parallel ports expand the card without merging wires or overflowing the diagram', () => {
    const edges = Array.from({ length: 20 }, (_, i) => wire('parallel-' + i, 'A', 'B'));
    const diagram = layoutStageDiagram(nodes.slice(0, 2), edges, orientation);
    checkGeometry(diagram, nodes.slice(0, 2), edges);
    assert.ok(diagram.edges.every(e => e.points.length === 2));
  });
}

test('blank or single block stages do not invent wires', () => {
  assert.deepEqual(layoutStageDiagram([], [wire('unused', 'A', 'Z')]), { width: 0, height: 0, nodes: [], edges: [] });
  const diagram = layoutStageDiagram(nodes.slice(0, 1), [wire('external', 'A', 'Z')]);
  assert.equal(diagram.nodes.length, 1);
  assert.deepEqual(diagram.nodes[0].ports, []);
  assert.deepEqual(diagram.edges, []);
});

test('a fully connected seven-block import keeps its directed routes visually separate', () => {
  const edges = nodes.flatMap((source, i) => nodes.map((target, j) =>
    wire(source.id + '-' + target.id, source.id, target.id, j > i ? 'forward' : 'feedback')));
  for (const orientation of ['horizontal', 'vertical']) {
    checkGeometry(layoutStageDiagram(nodes, edges, orientation), nodes, edges);
  }
});
