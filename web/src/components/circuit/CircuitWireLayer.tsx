import type { ActiveWiringDraft, CircuitEdge, CircuitNode } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { getPortAnchorPosition } from '../../lib/circuitEngine';
import { getSignalColor } from './blockIcons';

interface CircuitWireLayerProps {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  theme: ThemeMode;
  selectedEdgeId: string | null;
  activeEdgeIds: Set<string>;
  isRunning: boolean;
  wiringDraft: ActiveWiringDraft | null;
  onSelectEdge: (edgeId: string) => void;
}

interface ComputedPath {
  d: string;
  midX: number;
  midY: number;
  angleDeg: number;
}

function buildForwardBezier(x1: number, y1: number, x2: number, y2: number): ComputedPath {
  const dx = x2 - x1;
  const dist = Math.hypot(dx, y2 - y1);
  // Generous smooth horizontal tangent handle; bows gently if spanning multiple columns
  const handle = Math.max(58, Math.min(220, Math.abs(dx) * 0.46 + (dx < 0 ? 90 : 0)));
  const c1x = x1 + handle;
  const c1y = y1;
  const c2x = x2 - handle;
  const c2y = y2;
  const d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;

  // Cubic Bézier midpoint & tangent at t = 0.5
  const midX = 0.125 * (x1 + 3 * c1x + 3 * c2x + x2);
  const midY = 0.125 * (y1 + 3 * c1y + 3 * c2y + y2);
  const tx = 0.75 * (c1x - x1) + 1.5 * (c2x - c1x) + 0.75 * (x2 - c2x);
  const ty = 0.75 * (c1y - y1) + 1.5 * (c2y - c1y) + 0.75 * (y2 - c2y);
  const angleDeg = dist > 1 ? (Math.atan2(ty, tx) * 180) / Math.PI : 0;

  return { d, midX, midY, angleDeg };
}

function buildFeedbackArc(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  minNodeY: number,
): ComputedPath {
  // Elevated overhead smooth return arc arching gracefully above the breadboard
  const apexY = Math.min(y1, y2, minNodeY) - 86;
  const c1x = x1 + 64;
  const c1y = apexY;
  const c2x = x2 - 64;
  const c2y = apexY;
  const d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
  const midX = 0.125 * (x1 + 3 * c1x + 3 * c2x + x2);
  const midY = 0.125 * (y1 + 3 * c1y + 3 * c2y + y2);
  return { d, midX, midY, angleDeg: 180 };
}

export function CircuitWireLayer({
  nodes,
  edges,
  theme,
  selectedEdgeId,
  activeEdgeIds,
  isRunning,
  wiringDraft,
  onSelectEdge,
}: CircuitWireLayerProps) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const minNodeY =
    nodes.length > 0 ? Math.min(...nodes.map((n) => n.position.y)) : 64;

  return (
    <svg
      className="absolute inset-0 overflow-visible pointer-events-none"
      style={{ width: 1, height: 1 }}
      aria-label="Circuit signal wires"
    >
      <defs>
        <filter id="circuit-wire-glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="3.2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {edges.map((edge) => {
        const srcNode = nodeMap.get(edge.source);
        const dstNode = nodeMap.get(edge.target);
        if (!srcNode || !dstNode) return null;

        const p1 = getPortAnchorPosition(srcNode, edge.sourcePort, 'output');
        const p2 = getPortAnchorPosition(dstNode, edge.targetPort, 'input');
        const isFeedback = edge.kind === 'feedback';
        const isSelected = selectedEdgeId === edge.id;
        const isActive = activeEdgeIds.has(edge.id);

        const color = isFeedback
          ? theme === 'dark'
            ? '#E879F9'
            : '#C026D3'
          : getSignalColor(edge.signalType, theme);

        const geom = isFeedback
          ? buildFeedbackArc(p1.x, p1.y, p2.x, p2.y, minNodeY)
          : buildForwardBezier(p1.x, p1.y, p2.x, p2.y);

        // Subtle long-span opacity so multi-column wires never overpower nodes
        const span = Math.abs(p2.x - p1.x);
        const baseOpacity = isSelected || isActive ? 1 : isFeedback ? 0.92 : span > 450 ? 0.62 : 0.84;

        return (
          <g key={edge.id} className="group">
            {/* Soft ambient glow under selected or executing wires */}
            {(isSelected || isActive) && (
              <path
                d={geom.d}
                fill="none"
                stroke={color}
                strokeWidth={isSelected ? 7 : 6}
                strokeOpacity={theme === 'dark' ? 0.28 : 0.22}
                strokeLinecap="round"
              />
            )}

            {/* Main Signal Wire */}
            <path
              d={geom.d}
              fill="none"
              stroke={color}
              strokeWidth={isSelected ? 3 : isActive ? 2.8 : 2.15}
              strokeOpacity={baseOpacity}
              strokeLinecap="round"
              className={
                isFeedback
                  ? 'circuit-wire-feedback'
                  : isActive
                    ? 'circuit-wire-animated'
                    : ''
              }
            />

            {/* Directional Midpoint Chevron (forward wires) */}
            {!isFeedback && (
              <g
                transform={`translate(${geom.midX}, ${geom.midY}) rotate(${geom.angleDeg})`}
                opacity={isSelected || isActive ? 1 : 0.85}
              >
                <path
                  d="M -4 -3.6 L 3.8 0 L -4 3.6 Z"
                  fill={color}
                />
              </g>
            )}

            {/* Elevated Frosted Badge Pill on Feedback Loop Apex */}
            {isFeedback && (
              <g transform={`translate(${geom.midX}, ${geom.midY})`}>
                <rect
                  x={-68}
                  y={-12}
                  width={136}
                  height={24}
                  rx={12}
                  fill={theme === 'dark' ? 'rgba(15, 23, 42, 0.92)' : 'rgba(255, 255, 255, 0.96)'}
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

            {/* Animated Signal Particle Traveling Along Active Wire */}
            {isActive && isRunning && (
              <g filter="url(#circuit-wire-glow)">
                <circle r="4.5" fill={color}>
                  <animateMotion dur="0.55s" repeatCount="indefinite" path={geom.d} />
                </circle>
                <circle r="2.2" fill="#ffffff">
                  <animateMotion dur="0.55s" repeatCount="indefinite" path={geom.d} />
                </circle>
              </g>
            )}

            {/* Generous 14px Transparent Clickable Hit Target */}
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

      {/* Live Elastic Draft Wire while user is wiring */}
      {wiringDraft && (() => {
        const hovered = wiringDraft.hoveredTarget;
        const color = hovered
          ? !hovered.compatible
            ? theme === 'dark'
              ? '#FB7185'
              : '#E11D48'
            : hovered.wouldBeFeedback
              ? theme === 'dark'
                ? '#E879F9'
                : '#C026D3'
              : getSignalColor(wiringDraft.signalType, theme)
          : getSignalColor(wiringDraft.signalType, theme);

        const draftGeom = hovered?.wouldBeFeedback
          ? buildFeedbackArc(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
              minNodeY,
            )
          : buildForwardBezier(
              wiringDraft.startX,
              wiringDraft.startY,
              wiringDraft.cursorX,
              wiringDraft.cursorY,
            );

        return (
          <g>
            <path
              d={draftGeom.d}
              fill="none"
              stroke={color}
              strokeWidth={6}
              strokeOpacity={0.22}
              strokeLinecap="round"
            />
            <path
              d={draftGeom.d}
              fill="none"
              stroke={color}
              strokeWidth={2.5}
              strokeDasharray="6 4"
              strokeLinecap="round"
            />
            <circle
              cx={wiringDraft.cursorX}
              cy={wiringDraft.cursorY}
              r={4.5}
              fill={color}
            />
          </g>
        );
      })()}
    </svg>
  );
}
