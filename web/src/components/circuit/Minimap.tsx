import type React from 'react';
import type { CircuitEdge, CircuitNode } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { BLOCK_MAP, getCategoryColor, getSignalColor } from '../../lib/circuitCatalog';
import { getNodeHeight, NODE_WIDTH } from '../../lib/circuitEngine';

interface MinimapProps {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  viewport: { x: number; y: number; zoom: number };
  canvasSize: { width: number; height: number };
  selectedNodeId: string | null;
  theme: ThemeMode;
  onJumpToWorldPoint: (worldX: number, worldY: number) => void;
}

const MAP_W = 178;
const MAP_H = 110;
const PAD = 10;

export function Minimap({
  nodes,
  edges,
  viewport,
  canvasSize,
  selectedNodeId,
  theme,
  onJumpToWorldPoint,
}: MinimapProps) {
  if (nodes.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const n of nodes) {
    const h = getNodeHeight(n, false);
    minX = Math.min(minX, n.position.x);
    minY = Math.min(minY, n.position.y - 60);
    maxX = Math.max(maxX, n.position.x + NODE_WIDTH);
    maxY = Math.max(maxY, n.position.y + h);
  }

  // Include current viewport bounds in minimap projection
  const viewMinX = -viewport.x / viewport.zoom;
  const viewMinY = -viewport.y / viewport.zoom;
  const viewW = canvasSize.width / viewport.zoom;
  const viewH = canvasSize.height / viewport.zoom;

  minX = Math.min(minX, viewMinX) - 60;
  minY = Math.min(minY, viewMinY) - 60;
  maxX = Math.max(maxX, viewMinX + viewW) + 60;
  maxY = Math.max(maxY, viewMinY + viewH) + 60;

  const worldW = Math.max(400, maxX - minX);
  const worldH = Math.max(260, maxY - minY);
  const scale = Math.min((MAP_W - PAD * 2) / worldW, (MAP_H - PAD * 2) / worldH);

  const offsetX = PAD + (MAP_W - PAD * 2 - worldW * scale) / 2;
  const offsetY = PAD + (MAP_H - PAD * 2 - worldH * scale) / 2;

  const toMap = (wx: number, wy: number) => ({
    x: offsetX + (wx - minX) * scale,
    y: offsetY + (wy - minY) * scale,
  });

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const vpTopLeft = toMap(viewMinX, viewMinY);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const wx = minX + (mx - offsetX) / scale;
    const wy = minY + (my - offsetY) / scale;
    onJumpToWorldPoint(wx, wy);
  };

  return (
    <div
      data-circuit-minimap="true"
      className="absolute bottom-4 left-4 z-30 rounded-2xl circuit-floating-pill overflow-hidden select-none"
    >
      <svg
        width={MAP_W}
        height={MAP_H}
        onClick={handleClick}
        className="block cursor-crosshair"
        aria-label="Circuit Minimap"
      >
        {/* Scaled Wires */}
        {edges.map((edge) => {
          const s = nodeMap.get(edge.source);
          const t = nodeMap.get(edge.target);
          if (!s || !t) return null;
          const p1 = toMap(s.position.x + NODE_WIDTH, s.position.y + 70);
          const p2 = toMap(t.position.x, t.position.y + 70);
          return (
            <line
              key={edge.id}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke={
                edge.kind === 'feedback'
                  ? theme === 'dark'
                    ? '#E879F9'
                    : '#C026D3'
                  : getSignalColor(edge.signalType, theme)
              }
              strokeWidth={1.1}
              strokeOpacity={0.6}
              strokeDasharray={edge.kind === 'feedback' ? '2 2' : undefined}
            />
          );
        })}

        {/* Scaled Nodes */}
        {nodes.map((node) => {
          const def = BLOCK_MAP[node.blockType];
          const color = def ? getCategoryColor(def.category, theme) : '#38BDF8';
          const pos = toMap(node.position.x, node.position.y);
          const w = Math.max(10, NODE_WIDTH * scale);
          const h = Math.max(7, getNodeHeight(node, false) * scale);
          const isSelected = selectedNodeId === node.id;
          return (
            <rect
              key={node.id}
              x={pos.x}
              y={pos.y}
              width={w}
              height={h}
              rx={2.5}
              fill={color}
              fillOpacity={isSelected ? 0.95 : 0.65}
              stroke={isSelected ? '#FFFFFF' : 'none'}
              strokeWidth={1}
            />
          );
        })}

        {/* Viewport Indicator Box */}
        <rect
          x={vpTopLeft.x}
          y={vpTopLeft.y}
          width={Math.max(16, viewW * scale)}
          height={Math.max(12, viewH * scale)}
          rx={4}
          fill={theme === 'dark' ? 'rgba(56, 189, 248, 0.12)' : 'rgba(2, 132, 199, 0.1)'}
          stroke={theme === 'dark' ? '#38BDF8' : '#0284C7'}
          strokeWidth={1.25}
        />
      </svg>
    </div>
  );
}
