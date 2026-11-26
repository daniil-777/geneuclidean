import type { CircuitEdge, CircuitNode, WiringDraft } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { getSignalColor } from '../../lib/circuitCatalog';
import { getPortPosition } from '../../lib/circuitEngine';

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

function forwardBezier(x1: number, y1: number, x2: number, y2: number): WireGeom {
  const dx = x2 - x1;
  const handle = Math.max(54, Math.min(210, Math.abs(dx) * 0.45 + (dx < 0 ? 84 : 0)));
  const c1x = x1 + handle;
  const c1y = y1;
  const c2x = x2 - handle;
  const c2y = y2;
  const d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
  const midX = 0.125 * (x1 + 3 * c1x + 3 * c2x + x2);
  const midY = 0.125 * (y1 + 3 * c1y + 3 * c2y + y2);
  const tx = 0.75 * (c1x - x1) + 1.5 * (c2x - c1x) + 0.75 * (x2 - c2x);
  const ty = 0.75 * (c1y - y1) + 1.5 * (c2y - c1y) + 0.75 * (y2 - c2y);
  const angleDeg = Math.hypot(tx, ty) > 0.5 ? (Math.atan2(ty, tx) * 180) / Math.PI : 0;
  return { d, midX, midY, angleDeg };
}

function feedbackArc(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  minNodeY: number,
  staggerIdx: number,
): WireGeom {
  const apexY = Math.min(y1, y2, minNodeY) - 76 - staggerIdx * 34;
  const c1x = x1 + 60;
  const c1y = apexY;
  const c2x = x2 - 60;
  const c2y = apexY;
  const d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
  const midX = 0.125 * (x1 + 3 * c1x + 3 * c2x + x2);
  const midY = 0.125 * (y1 + 3 * c1y + 3 * c2y + y2);
  return { d, midX, midY, angleDeg: 180 };
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
  const minNodeY = nodes.length > 0 ? Math.min(...nodes.map((n) => n.position.y)) : 64;
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
      </defs>

      {edges.map((edge) => {
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
          ? feedbackArc(p1.x, p1.y, p2.x, p2.y, minNodeY, fbCounter++)
          : forwardBezier(p1.x, p1.y, p2.x, p2.y);

        const span = Math.abs(p2.x - p1.x);
        const opacity = isSelected || isActive ? 1 : isFeedback ? 0.94 : span > 480 ? 0.62 : 0.85;

        return (
          <g key={edge.id} data-circuit-edge={edge.id}>
            {/* Ambient Highlight Halo */}
            {(isSelected || isActive) && (
              <path
                d={geom.d}
                fill="none"
                stroke={color}
                strokeWidth={isSelected ? 7.5 : 6}
                strokeOpacity={theme === 'dark' ? 0.28 : 0.2}
                strokeLinecap="round"
              />
            )}

            {/* Main Signal Trace */}
            <path
              d={geom.d}
              fill="none"
              stroke={color}
              strokeWidth={isSelected ? 3 : isActive ? 2.75 : 2.15}
              strokeOpacity={opacity}
              strokeDasharray={isFeedback ? '6 4' : undefined}
              strokeLinecap="round"
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
                opacity={isSelected || isActive ? 1 : 0.88}
              >
                <path d="M -4.2 -3.6 L 4 0 L -4.2 3.6 Z" fill={color} />
              </g>
            )}

            {/* Frosted Pill Badge on DMTA Feedback Loop Apex */}
            {isFeedback && (
              <g transform={`translate(${geom.midX}, ${geom.midY})`}>
                <rect
                  x={-72}
                  y={-12}
                  width={144}
                  height={24}
                  rx={12}
                  fill={theme === 'dark' ? 'rgba(13, 19, 34, 0.94)' : 'rgba(255, 255, 255, 0.96)'}
                  stroke={color}
                  strokeWidth={1.25}
                />
                <text
                  x={0}
                  y={3.5}
                  textAnchor="middle"
                  fill={color}
                  fontSize="10.5"
                  fontWeight="600"
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
          ? feedbackArc(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
              minNodeY,
              0,
            )
          : forwardBezier(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
            );

        return (
          <g>
            <path
              d={g.d}
              fill="none"
              stroke={color}
              strokeWidth={6}
              strokeOpacity={0.22}
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
            <circle cx={wiringDraft.cursorX} cy={wiringDraft.cursorY} r={4.5} fill={color} />
          </g>
        );
      })()}
    </svg>
  );
}
