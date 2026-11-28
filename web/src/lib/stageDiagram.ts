import type { CircuitEdge, CircuitNode } from '../types/circuit';

export type DiagramOrientation = 'horizontal' | 'vertical';
export type DiagramSide = 'top' | 'right' | 'bottom' | 'left';
export interface DiagramPoint { x: number; y: number }
export interface DiagramPort extends DiagramPoint {
  edgeId: string;
  portId: string;
  direction: 'input' | 'output';
  side: DiagramSide;
}
export interface DiagramNode extends DiagramPoint {
  id: string;
  width: number;
  height: number;
  ports: DiagramPort[];
}
export interface DiagramEdge extends CircuitEdge {
  points: DiagramPoint[];
  /** Visible centerline, with small breaks where a custom graph forces a crossing. */
  path: string;
  /** Continuous centerline for pointer targets and the target arrowhead. */
  hitPath: string;
}
export interface StageDiagram {
  width: number;
  height: number;
  nodes: DiagramNode[];
  edges: DiagramEdge[];
}

const padding = 14;
const laneGap = 18;
const socketGap = 20;
const crossingGap = 4;
type Gutter = 'left' | 'right';

interface Route {
  edge: CircuitEdge;
  source: number;
  target: number;
  gutter?: Gutter;
  lane: number;
  start?: DiagramPoint;
  end?: DiagramPoint;
}
interface Attachment {
  route: Route;
  direction: 'input' | 'output';
  upstream: boolean;
}

function compareId(a: CircuitEdge, b: CircuitEdge): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

// Sweep backwards so a route enclosing a shorter one gets an outer lane.
// Intervals sharing even one block get separate lanes and separate sockets.
function assignLanes(routes: Route[]): number {
  const start = (r: Route) => Math.min(r.source, r.target);
  const end = (r: Route) => Math.max(r.source, r.target);
  routes.sort((a, b) => start(b) - start(a) || end(a) - end(b) || compareId(a.edge, b.edge));
  const firstOccupied: number[] = [];
  for (const route of routes) {
    const freeLane = firstOccupied.findIndex(startIndex => end(route) < startIndex);
    route.lane = freeLane < 0 ? firstOccupied.length : freeLane;
    firstOccupied[route.lane] = start(route);
  }
  return firstOccupied.length;
}

interface Rail { x: number; spans: { low: number; high: number }[] }
function occupied(rail: Rail, y: number): boolean {
  // Disjoint spans in one reusable rail are sorted by their start coordinate.
  let left = 0;
  let right = rail.spans.length;
  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (rail.spans[mid].low < y) left = mid + 1;
    else right = mid;
  }
  return left > 0 && rail.spans[left - 1].high > y;
}

/** Lay out existing blocks and their internal wires in the stage's dependency order.
 * Coordinates are independent of the editable canvas and expressed in CSS pixels. */
export function layoutStageDiagram(
  nodes: readonly CircuitNode[],
  edges: readonly CircuitEdge[],
  orientation: DiagramOrientation = 'vertical',
): StageDiagram {
  if (!nodes.length) return { width: 0, height: 0, nodes: [], edges: [] };
  // Route in vertical coordinates first. Transpose the finished diagram for desktop.
  const horizontal = orientation === 'horizontal';
  const point = (x: number, y: number): DiagramPoint => horizontal ? { x: y, y: x } : { x, y };
  const transpose: Record<DiagramSide, DiagramSide> = { top: 'left', right: 'bottom', bottom: 'right', left: 'top' };
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const routes: Route[] = [];
  const adjacent = nodes.map((): Route[] => []);
  const sideRoutes: Record<Gutter, Route[]> = { left: [], right: [] };
  const sockets = nodes.map((): Record<Gutter, Attachment[]> => ({ left: [], right: [] }));

  for (const edge of [...edges].sort(compareId)) {
    const source = index.get(edge.source);
    const target = index.get(edge.target);
    if (source === undefined || target === undefined) continue;
    const route: Route = { edge, source, target, lane: 0 };
    routes.push(route);
    if (edge.kind === 'forward' && target === source + 1) {
      adjacent[source].push(route);
    } else {
      const gutter = edge.kind === 'feedback' || target < source ? 'left' : 'right';
      route.gutter = gutter;
      sideRoutes[gutter].push(route);
      sockets[source][gutter].push({ route, direction: 'output', upstream: target < source });
      sockets[target][gutter].push({ route, direction: 'input', upstream: source <= target });
    }
  }

  const leftLanes = assignLanes(sideRoutes.left);
  const rightLanes = assignLanes(sideRoutes.right);
  const maxParallel = adjacent.reduce((max, group) => Math.max(max, group.length), 0);
  const breadth = Math.max(horizontal ? 94 : 212, (maxParallel - 1) * socketGap + 36);
  const cardX = padding + leftLanes * laneGap;
  const diagramBreadth = cardX + breadth + rightLanes * laneGap + padding;
  const cardGap = horizontal ? 64 : 52;
  let cursor = padding;
  const cards = nodes.map((node, i) => {
    const count = Math.max(sockets[i].left.length, sockets[i].right.length);
    const length = Math.max(horizontal ? 210 : 84, (count - 1) * socketGap + 32);
    const y = cursor;
    cursor += length + cardGap;
    const laidOut: DiagramNode = { id: node.id, ...point(cardX, y),
      width: horizontal ? length : breadth, height: horizontal ? breadth : length, ports: [] };
    return { y, length, laidOut };
  });

  function attach(route: Route, direction: 'input' | 'output', side: DiagramSide, x: number, y: number) {
    const source = direction === 'output';
    if (source) route.start = { x, y };
    else route.end = { x, y };
    cards[source ? route.source : route.target].laidOut.ports.push({
      edgeId: route.edge.id, portId: source ? route.edge.sourcePort : route.edge.targetPort,
      direction, side: horizontal ? transpose[side] : side, ...point(x, y),
    });
  }
  for (const [i, card] of cards.entries()) {
    const group = adjacent[i];
    for (const [j, route] of group.entries()) {
      const x = cardX + breadth / 2 + (j - (group.length - 1) / 2) * socketGap;
      attach(route, 'output', 'bottom', x, card.y + card.length);
      attach(route, 'input', 'top', x, cards[i + 1].y);
    }
    for (const gutter of ['left', 'right'] as const) {
      const attachments = sockets[i][gutter];
      // Put upstream sockets above downstream sockets, and fan parallel wires without crossing.
      attachments.sort((a, b) => Number(b.upstream) - Number(a.upstream) ||
        (a.upstream ? a.route.lane - b.route.lane : b.route.lane - a.route.lane));
      for (const [j, { route, direction }] of attachments.entries()) {
        const y = card.y + card.length / 2 + (j - (attachments.length - 1) / 2) * socketGap;
        attach(route, direction, gutter, gutter === 'left' ? cardX : cardX + breadth, y);
      }
    }
  }

  function rails(count: number, gutter: Gutter): Rail[] {
    return Array.from({ length: count }, (_, lane) => ({
      x: gutter === 'left' ? cardX - (lane + 1) * laneGap : cardX + breadth + (lane + 1) * laneGap,
      spans: [],
    }));
  }
  const gutters = { left: rails(leftLanes, 'left'), right: rails(rightLanes, 'right') };
  for (const gutter of ['left', 'right'] as const) {
    for (const route of sideRoutes[gutter]) {
      const a = route.start!.y, b = route.end!.y;
      gutters[gutter][route.lane].spans.push({ low: Math.min(a, b), high: Math.max(a, b) });
    }
    for (const rail of gutters[gutter]) rail.spans.sort((a, b) => a.low - b.low);
  }

  const command = (cmd: 'M' | 'L', p: DiagramPoint) => cmd + p.x + ' ' + p.y;
  const diagramEdges = routes.map(route => {
    const start = route.start!, end = route.end!;
    const virtual = [start];
    if (route.gutter) {
      const x = gutters[route.gutter][route.lane].x;
      virtual.push({ x, y: start.y }, { x, y: end.y });
    }
    virtual.push(end);
    const points = virtual.map(p => point(p.x, p.y));
    const hitPath = points.map((p, i) => command(i ? 'L' : 'M', p)).join(' ');
    const path = [command('M', points[0])];
    for (let i = 1; i < virtual.length; i++) {
      const a = virtual[i - 1], b = virtual[i];
      if (route.gutter && a.y === b.y && route.lane > 0) {
        const direction = Math.sign(b.x - a.x);
        // Only a nearer rail can intersect this horizontal leg. Leave a gap in this
        // wire at each intersection so a crossing cannot be read as a junction.
        const crossed = gutters[route.gutter].slice(0, route.lane).filter(rail => occupied(rail, a.y));
        if (i === virtual.length - 1) crossed.reverse();
        for (const { x } of crossed) {
          path.push(command('L', point(x - direction * crossingGap, a.y)),
            command('M', point(x + direction * crossingGap, a.y)));
        }
      }
      path.push(command('L', points[i]));
    }
    return { ...route.edge, points, hitPath, path: path.join(' ') };
  });
  const length = cursor - cardGap + padding;
  return { width: horizontal ? length : diagramBreadth, height: horizontal ? diagramBreadth : length,
    nodes: cards.map(card => card.laidOut), edges: diagramEdges };
}
