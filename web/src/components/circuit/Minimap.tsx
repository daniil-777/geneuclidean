import type { CircuitEdge, CircuitNode } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { BLOCK_CATALOG_MAP } from '../../lib/circuitCatalog';
import { NODE_WIDTH, estimateNodeHeight } from '../../lib/circuitEngine';
import { getCategoryColor } from './blockIcons';

interface MinimapProps {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  viewport: { x: number; y: number; zoom: number };
  canvasSize: { width: number; height: number };
  theme: ThemeMode;
  selectedNodeId: string | null;
  onJumpToWorld: (worldX: number, worldY: number) => void;
}

const MAP_W = 176;
const MAP_H = 108;
const PAD = 10;

export function Minimap({
  nodes,
  edges,
  viewport,
  canvasSize,
  theme,
  selectedNodeId,
  onJumpToWorld,
}: MinimapProps) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const n of nodes) {
    const h = estimateNodeHeight(n, false);
    minX = Math.min(minX, n.position.x);
    minY = Math.min(minY, n.position.y);
    maxX = Math.max(maxX, n.position.x + NODE_WIDTH);
    maxY = Math.max(maxY, n.position.y + h);
  }

  // Include current viewport bounds so indicator stays visible
  const vpLeft = -viewport.x / viewport.zoom;
  const vpTop = -viewport.y / viewport.zoom;
  const vpW = canvasSize.width / viewport.zoom;
  const vpH = canvasSize.height / viewport.zoom;

  minX = Math.min(minX, vpLeft) - 80;
  minY = Math.min(minY, vpTop) - 80;
  maxX = Math.max(maxX, vpLeft + vpW) + 80;
  maxY = Math.max(maxY, vpTop + vpH) + 80;

  const spanX = Math.max(400, maxX - minX);
  const spanY = Math.max(260, maxY - minY);
  const scale = Math.min((MAP_W - PAD * 2) / spanX, (MAP_H - PAD * 2) / spanY);
  const offsetX = (MAP_W - spanX * scale) / 2;
  const offsetY = (MAP_H - spanY * scale) / 2;

  const toMapX = (wx: number) => offsetX + (wx - minX) * scale;
  const toMapY = (wy: number) => offsetY + (wy - minY) * scale;

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const wx = minX + (mx - offsetX) / scale;
    const wy = minY + (my - offsetY) / scale;
    onJumpToWorld(wx, wy);
  };

  return (
    <div className="circuit-floating-pill rounded-2xl p-1.5 overflow-hidden">
      <svg
        width={MAP_W}
        height={MAP_H}
        onClick={handleClick}
        className="block rounded-xl cursor-pointer"
        aria-label="Circuit Minimap"
      >
        {/* Wires */}
        {edges.map((edge) => {
          const s = nodeMap.get(edge.source);
          const t = nodeMap.get(edge.target);
          if (!s || !t) return null;
          const x1 = toMapX(s.position.x + NODE_WIDTH);
          const y1 = toMapY(s.position.y + 44);
          const x2 = toMapX(t.position.x);
          const y2 = toMapY(t.position.y + 44);
          return (
            <line
              key={edge.id}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={
                edge.kind === 'feedback'
                  ? theme === 'dark'
                    ? '#E879F9'
                    : '#C026D3'
                  : theme === 'dark'
                    ? 'rgba(148,163,184,0.45)'
                    : 'rgba(15,23,42,0.3)'
              }
              strokeWidth={edge.kind === 'feedback' ? 1.2 : 0.9}
              strokeDasharray={edge.kind === 'feedback' ? '2 2' : undefined}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((node) => {
          const def = BLOCK_CATALOG_MAP[node.blockType];
          const color = def ? getCategoryColor(def.category, theme) : '#38BDF8';
          const h = estimateNodeHeight(node, false);
          const isSel = node.id === selectedNodeId;
          return (
            <rect
              key={node.id}
              x={toMapX(node.position.x)}
              y={toMapY(node.position.y)}
              width={Math.max(8, NODE_WIDTH * scale)}
              height={Math.max(5, h * scale)}
              rx={2}
              fill={color}
              fillOpacity={isSel ? 0.95 : 0.65}
              stroke={isSel ? (theme === 'dark' ? '#FFFFFF' : '#0F172A') : 'none'}
              strokeWidth={isSel ? 1.2 : 0}
            />
          );
        })}

        {/* Active Viewport Frame */}
        <rect
          x={toMapX(vpLeft)}
          y={toMapY(vpTop)}
          width={Math.max(12, vpW * scale)}
          height={Math.max(10, vpH * scale)}
          rx={3}
          fill={theme === 'dark' ? 'rgba(56,189,248,0.12)' : 'rgba(2,132,199,0.1)'}
          stroke={theme === 'dark' ? '#38BDF8' : '#0284C7'}
          strokeWidth={1.2}
        />
      </svg>
    </div>
  );
}
