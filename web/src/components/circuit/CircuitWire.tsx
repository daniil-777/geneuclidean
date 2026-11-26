import type { CircuitEdge, CircuitNode, WiringDraft } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { getSignalColor } from '../../lib/circuitCatalog';
import { getNodeHeight, getPortPosition, NODE_WIDTH } from '../../lib/circuitEngine';

interface CircuitWireProps {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  theme: ThemeMode;
  selectedEdgeId: string | null;
  activeEdgeIds: Set<string>;
  isRunning: boolean;
  wiringDraft: WiringDraft | null;
  onSelectEdge: (edgeId: string) => void;
}

interface WireGeom {
  d: string;
  midX: number;
  midY: number;
  angleDeg: number;
}

interface Rect {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

function buildNodeRects(nodes: CircuitNode[], pad = 8): Rect[] {
  return nodes.map((n) => ({
    id: n.id,
    x1: n.position.x - pad,
    y1: n.position.y - pad,
    x2: n.position.x + NODE_WIDTH + pad,
    y2: n.position.y + getNodeHeight(n, false) + pad,
  }));
}

function pointInRect(x: number, y: number, r: Rect): boolean {
  return x >= r.x1 && x <= r.x2 && y >= r.y1 && y <= r.y2;
}

function segmentIntersectsRect(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  r: Rect,
): boolean {
  // Fast AABB test for horizontal / vertical PCB segments
  if (Math.abs(ay - by) < 0.5) {
    const lo = Math.min(ax, bx);
    const hi = Math.max(ax, bx);
    return ay >= r.y1 && ay <= r.y2 && hi >= r.x1 && lo <= r.x2;
  }
  if (Math.abs(ax - bx) < 0.5) {
    const lo = Math.min(ay, by);
    const hi = Math.max(ay, by);
    return ax >= r.x1 && ax <= r.x2 && hi >= r.y1 && lo <= r.y2;
  }
  for (let i = 1; i <= 12; i += 1) {
    const t = i / 13;
    if (pointInRect(ax + (bx - ax) * t, ay + (by - ay) * t, r)) return true;
  }
  return false;
}

function bezierHitsAny(
  x1: number,
  y1: number,
  c1x: number,
  c1y: number,
  c2x: number,
  c2y: number,
  x2: number,
  y2: number,
  obstacles: Rect[],
): boolean {
  for (let i = 1; i <= 16; i += 1) {
    const t = i / 17;
    const u = 1 - t;
    const x = u * u * u * x1 + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * x2;
    const y = u * u * u * y1 + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * y2;
    for (const r of obstacles) {
      if (pointInRect(x, y, r)) return true;
    }
  }
  return false;
}

function buildRoundedOrthogonalPath(points: [number, number][], radius = 12): string {
  if (points.length < 2) return '';
  const cmds: string[] = [`M ${points[0][0]} ${points[0][1]}`];
  for (let i = 1; i < points.length - 1; i += 1) {
    const [px, py] = points[i - 1];
    const [cx, cy] = points[i];
    const [nx, ny] = points[i + 1];
    const d1 = Math.hypot(cx - px, cy - py);
    const d2 = Math.hypot(nx - cx, ny - cy);
    const r = Math.min(radius, d1 * 0.45, d2 * 0.45);
    if (r < 1) {
      cmds.push(`L ${cx} ${cy}`);
      continue;
    }
    const sx = cx - ((cx - px) / d1) * r;
    const sy = cy - ((cy - py) / d1) * r;
    const ex = cx + ((nx - cx) / d2) * r;
    const ey = cy + ((ny - cy) / d2) * r;
    cmds.push(`L ${sx} ${sy}`);
    cmds.push(`Q ${cx} ${cy} ${ex} ${ey}`);
  }
  const last = points[points.length - 1];
  cmds.push(`L ${last[0]} ${last[1]}`);
  return cmds.join(' ');
}

/**
 * Finds the clearest horizontal bus corridor Y-coordinate (inter-row corridor, top rail, or bottom rail)
 * guaranteed free of intermediate node bodies.
 */
function findCleanBusY(
  xExit: number,
  y1: number,
  xEnter: number,
  y2: number,
  obstacles: Rect[],
  allRects: Rect[],
  laneIndex: number,
): number {
  const minAllY = allRects.length ? Math.min(...allRects.map((r) => r.y1)) : 0;
  const maxAllY = allRects.length ? Math.max(...allRects.map((r) => r.y2)) : 400;

  // Detect natural inter-row gap (between upper and lower node bands)
  const sortedByTop = [...allRects].sort((a, b) => a.y1 - b.y1);
  const gapMidpoints: number[] = [];
  for (let i = 0; i < sortedByTop.length - 1; i += 1) {
    // Max bottom of nodes starting at or above sortedByTop[i].y1
    const upperMaxBottom = Math.max(
      ...allRects.filter((r) => r.y1 <= sortedByTop[i].y1 + 40).map((r) => r.y2),
    );
    const lowerMinTop = Math.min(
      ...allRects.filter((r) => r.y1 > sortedByTop[i].y1 + 40).map((r) => r.y1),
    );
    if (Number.isFinite(lowerMinTop) && lowerMinTop - upperMaxBottom >= 24) {
      const mid = (upperMaxBottom + lowerMinTop) / 2;
      if (!gapMidpoints.some((g) => Math.abs(g - mid) < 12)) {
        gapMidpoints.push(mid);
      }
    }
  }

  const stagger = ((laneIndex % 5) - 2) * 10;
  const candidates: number[] = [
    ...gapMidpoints.map((g) => g + stagger),
    minAllY - 22 - (laneIndex % 4) * 11,
    maxAllY + 22 + (laneIndex % 4) * 11,
  ];

  // Score each candidate rail by zero collisions first, then shortest Manhattan travel
  let bestY = candidates[0] ?? (y1 + y2) / 2;
  let bestScore = Infinity;

  for (const candY of candidates) {
    let hits = 0;
    for (const r of obstacles) {
      if (
        segmentIntersectsRect(xExit, y1, xExit, candY, r) ||
        segmentIntersectsRect(xExit, candY, xEnter, candY, r) ||
        segmentIntersectsRect(xEnter, candY, xEnter, y2, r)
      ) {
        hits += 1;
      }
    }
    const travel = Math.abs(candY - y1) + Math.abs(y2 - candY);
    const score = hits * 10000 + travel;
    if (score < bestScore) {
      bestScore = score;
      bestY = candY;
    }
  }
  return bestY;
}

/**
 * Obstacle-aware Apple PCB wire router:
 * - Adjacent unobstructed forward connections use silky horizontal cubic Bézier curves.
 * - Multi-column bypasses & inter-row transitions route cleanly through open inter-row or outer
 *   bus channels with rounded 12px corners and ZERO node body intersections.
 */
function routeForwardWire(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  allRects: Rect[],
  srcId: string,
  dstId: string,
  laneIndex: number,
): WireGeom {
  const dx = x2 - x1;
  const obstacles = allRects.filter((r) => r.id !== srcId && r.id !== dstId);

  // Test clean direct cubic Bézier first when flowing left-to-right
  if (dx > 16) {
    const handle = Math.max(44, Math.min(168, dx * 0.44));
    const c1x = x1 + handle;
    const c1y = y1;
    const c2x = x2 - handle;
    const c2y = y2;
    if (!bezierHitsAny(x1, y1, c1x, c1y, c2x, c2y, x2, y2, obstacles)) {
      const d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
      const midX = 0.125 * (x1 + 3 * c1x + 3 * c2x + x2);
      const midY = 0.125 * (y1 + 3 * c1y + 3 * c2y + y2);
      const tx = 0.75 * (c1x - x1) + 1.5 * (c2x - c1x) + 0.75 * (x2 - c2x);
      const ty = 0.75 * (c1y - y1) + 1.5 * (c2y - c1y) + 0.75 * (y2 - c2y);
      const angleDeg = Math.hypot(tx, ty) > 0.5 ? (Math.atan2(ty, tx) * 180) / Math.PI : 0;
      return { d, midX, midY, angleDeg };
    }
  }

  // Otherwise route smoothly along the clearest obstacle-free PCB bus channel
  const stub = 18 + (laneIndex % 3) * 4;
  const xExit = x1 + stub;
  const xEnter = x2 - stub;
  const busY = findCleanBusY(xExit, y1, xEnter, y2, obstacles, allRects, laneIndex);

  const pts: [number, number][] = [
    [x1, y1],
    [xExit, y1],
    [xExit, busY],
    [xEnter, busY],
    [xEnter, y2],
    [x2, y2],
  ];
  const d = buildRoundedOrthogonalPath(pts, 12);
  const midX = (xExit + xEnter) / 2;
  const angleDeg = xEnter >= xExit ? 0 : 180;
  return { d, midX, midY: busY, angleDeg };
}

/**
 * Elevated obstacle-free DMTA feedback loop router:
 * Exits cleanly into the inter-column gutter, rises above all nodes to apexY, sweeps left with
 * a crisp frosted `↻ DMTA Feedback Loop` pill, and drops cleanly down the target's left gutter.
 */
function routeFeedbackWire(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  allRects: Rect[],
  srcId: string,
  dstId: string,
  staggerIdx: number,
): WireGeom {
  const minAllY = allRects.length ? Math.min(...allRects.map((r) => r.y1)) : 40;
  const apexY = minAllY - 46 - staggerIdx * 32;
  const obstacles = allRects.filter((r) => r.id !== srcId && r.id !== dstId);

  // Choose clean exit & entry vertical gutters that clear every obstacle
  let xExit = x1 + 24;
  if (obstacles.some((r) => segmentIntersectsRect(xExit, y1, xExit, apexY, r))) {
    const maxRight = Math.max(...allRects.map((r) => r.x2), x1);
    xExit = maxRight + 22;
  }
  let xEnter = x2 - 24;
  if (obstacles.some((r) => segmentIntersectsRect(xEnter, apexY, xEnter, y2, r))) {
    const minLeft = Math.min(...allRects.map((r) => r.x1), x2);
    xEnter = minLeft - 22;
  }

  const pts: [number, number][] = [
    [x1, y1],
    [xExit, y1],
    [xExit, apexY],
    [xEnter, apexY],
    [xEnter, y2],
    [x2, y2],
  ];
  const d = buildRoundedOrthogonalPath(pts, 16);
  return { d, midX: (xExit + xEnter) / 2, midY: apexY, angleDeg: 180 };
}

export function CircuitWire({
  nodes,
  edges,
  theme,
  selectedEdgeId,
  activeEdgeIds,
  isRunning,
  wiringDraft,
  onSelectEdge,
}: CircuitWireProps) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const allRects = buildNodeRects(nodes, 7);
  let fbCounter = 0;

  return (
    <svg
      className="absolute top-0 left-0 overflow-visible pointer-events-none z-0"
      width={1}
      height={1}
      aria-label="Circuit signal wires"
    >
      <defs>
        <filter id="ge-signal-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="ge-pill-shadow" x="-20%" y="-40%" width="140%" height="200%">
          <feDropShadow
            dx="0"
            dy="3"
            stdDeviation="4"
            floodColor={theme === 'dark' ? '#000000' : '#0F172A'}
            floodOpacity={theme === 'dark' ? '0.55' : '0.14'}
          />
        </filter>
      </defs>

      {edges.map((edge, edgeIdx) => {
        const sNode = nodeMap.get(edge.source);
        const tNode = nodeMap.get(edge.target);
        if (!sNode || !tNode) return null;

        const p1 = getPortPosition(sNode, edge.sourcePort, 'output');
        const p2 = getPortPosition(tNode, edge.targetPort, 'input');
        const isFeedback = edge.kind === 'feedback';
        const isSelected = selectedEdgeId === edge.id;
        const isActive = activeEdgeIds.has(edge.id);

        const color = isFeedback
          ? theme === 'dark'
            ? '#E879F9'
            : '#C026D3'
          : getSignalColor(edge.signalType, theme);

        const geom = isFeedback
          ? routeFeedbackWire(p1.x, p1.y, p2.x, p2.y, allRects, edge.source, edge.target, fbCounter++)
          : routeForwardWire(p1.x, p1.y, p2.x, p2.y, allRects, edge.source, edge.target, edgeIdx);

        const span = Math.abs(p2.x - p1.x);
        const opacity = isSelected || isActive ? 1 : isFeedback ? 0.96 : span > 420 ? 0.78 : 0.88;

        return (
          <g key={edge.id} data-circuit-edge={edge.id}>
            {/* Ambient Highlight Halo */}
            {(isSelected || isActive) && (
              <path
                d={geom.d}
                fill="none"
                stroke={color}
                strokeWidth={isSelected ? 7.5 : 6}
                strokeOpacity={theme === 'dark' ? 0.3 : 0.22}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            )}

            {/* Main Signal Trace */}
            <path
              d={geom.d}
              fill="none"
              stroke={color}
              strokeWidth={isSelected ? 3 : isActive ? 2.75 : 2.1}
              strokeOpacity={opacity}
              strokeDasharray={isFeedback ? '6 4' : undefined}
              strokeLinecap="round"
              strokeLinejoin="round"
              className={
                isFeedback
                  ? 'circuit-wire-feedback-flow'
                  : isActive
                    ? 'circuit-wire-animated'
                    : ''
              }
            />

            {/* Directional Midpoint Arrowhead (Forward Wires) */}
            {!isFeedback && (
              <g
                transform={`translate(${geom.midX}, ${geom.midY}) rotate(${geom.angleDeg})`}
                opacity={isSelected || isActive ? 1 : 0.9}
              >
                <path d="M -4.5 -3.6 L 4.2 0 L -4.5 3.6 Z" fill={color} />
              </g>
            )}

            {/* Crisp Frosted Pill Badge on DMTA Feedback Loop Apex */}
            {isFeedback && (
              <g transform={`translate(${geom.midX}, ${geom.midY})`} filter="url(#ge-pill-shadow)">
                <rect
                  x={-84}
                  y={-14}
                  width={168}
                  height={28}
                  rx={14}
                  fill={theme === 'dark' ? 'rgba(13, 19, 34, 0.96)' : 'rgba(255, 255, 255, 0.98)'}
                  stroke={color}
                  strokeWidth={1.5}
                />
                <text
                  x={0}
                  y={4}
                  textAnchor="middle"
                  fill={color}
                  fontSize="11.5"
                  fontWeight="700"
                  fontFamily="var(--font-mono)"
                >
                  ↻ DMTA Feedback Loop
                </text>
              </g>
            )}

            {/* Luminous Animated Signal Particle on Execution Wave */}
            {isActive && isRunning && (
              <g filter="url(#ge-signal-glow)">
                <circle r="4.5" fill={color}>
                  <animateMotion dur="0.52s" repeatCount="indefinite" path={geom.d} />
                </circle>
                <circle r="2" fill="#FFFFFF">
                  <animateMotion dur="0.52s" repeatCount="indefinite" path={geom.d} />
                </circle>
              </g>
            )}

            {/* Wide 14px Clickable Hit Target */}
            <path
              d={geom.d}
              fill="none"
              stroke="transparent"
              strokeWidth={14}
              className="pointer-events-auto cursor-pointer"
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelectEdge(edge.id);
              }}
            />
          </g>
        );
      })}

      {/* Live Elastic Draft Wire */}
      {wiringDraft && (() => {
        const h = wiringDraft.hoveredTarget;
        const color = h
          ? !h.compatible
            ? theme === 'dark'
              ? '#FB7185'
              : '#E11D48'
            : h.willBeFeedback
              ? theme === 'dark'
                ? '#E879F9'
                : '#C026D3'
              : getSignalColor(wiringDraft.signalType, theme)
          : getSignalColor(wiringDraft.signalType, theme);

        const g = h?.willBeFeedback
          ? routeFeedbackWire(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
              allRects,
              wiringDraft.sourceNodeId,
              h.nodeId,
              0,
            )
          : routeForwardWire(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
              [],
              '',
              '',
              0,
            );

        return (
          <g>
            <path
              d={g.d}
              fill="none"
              stroke={color}
              strokeWidth={6}
              strokeOpacity={0.24}
              strokeLinecap="round"
            />
            <path
              d={g.d}
              fill="none"
              stroke={color}
              strokeWidth={2.5}
              strokeDasharray="6 4"
              strokeLinecap="round"
            />
            <circle cx={wiringDraft.cursorX} cy={wiringDraft.cursorY} r={5} fill={color} />
          </g>
        );
      })()}
    </svg>
  );
}
