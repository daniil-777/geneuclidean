import { Check, AlertTriangle, RotateCcw } from 'lucide-react';
import type {
  ActiveWiringDraft,
  CircuitNode,
  ExecutionStepTelemetry,
  NodeRuntimeStatus,
  PortSpec,
} from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { BLOCK_CATALOG_MAP, formatNodeParamSummary } from '../../lib/circuitCatalog';
import {
  NODE_HEADER_HEIGHT,
  NODE_TELEMETRY_FOOTER_HEIGHT,
  NODE_WIDTH,
  PORT_AREA_BOTTOM_PAD,
  PORT_AREA_TOP_PAD,
  PORT_ROW_HEIGHT,
} from '../../lib/circuitEngine';
import { getBlockIcon, getCategoryColor, getSignalColor } from './blockIcons';

interface CircuitNodeCardProps {
  node: CircuitNode;
  theme: ThemeMode;
  selected: boolean;
  hasError: boolean;
  runtimeStatus: NodeRuntimeStatus;
  latestTelemetry?: ExecutionStepTelemetry;
  wiringDraft: ActiveWiringDraft | null;
  isPortCompatible: (nodeId: string, portId: string) => { compatible: boolean; wouldBeFeedback: boolean; remediation?: string };
  onNodePointerDown: (e: React.PointerEvent, nodeId: string) => void;
  onStartWiring: (e: React.PointerEvent, nodeId: string, port: PortSpec) => void;
  onCompleteWiring: (e: React.PointerEvent, nodeId: string, port: PortSpec) => void;
  onHoverInputPort: (nodeId: string, port: PortSpec | null) => void;
}

export function CircuitNodeCard({
  node,
  theme,
  selected,
  hasError,
  runtimeStatus,
  latestTelemetry,
  wiringDraft,
  isPortCompatible,
  onNodePointerDown,
  onStartWiring,
  onCompleteWiring,
  onHoverInputPort,
}: CircuitNodeCardProps) {
  const def = BLOCK_CATALOG_MAP[node.blockType];
  if (!def) return null;

  const Icon = getBlockIcon(node.blockType);
  const catColor = getCategoryColor(def.category, theme);
  const paramSummary = formatNodeParamSummary(node);
  const portRowCount = Math.max(1, def.inputs.length, def.outputs.length);
  const isExecuting = runtimeStatus === 'running';

  const statusDotClass =
    runtimeStatus === 'running'
      ? 'bg-cyan-400 animate-ping'
      : runtimeStatus === 'ok'
        ? 'bg-emerald-500'
        : runtimeStatus === 'simulated_fallback'
          ? 'bg-amber-400'
          : runtimeStatus === 'error' || hasError
            ? 'bg-rose-500'
            : 'bg-slate-400/45';

  return (
    <div
      data-circuit-node={node.id}
      onPointerDown={(e) => onNodePointerDown(e, node.id)}
      style={{
        transform: `translate3d(${node.position.x}px, ${node.position.y}px, 0)`,
        width: NODE_WIDTH,
        boxShadow: selected
          ? 'var(--circuit-node-shadow-selected)'
          : 'var(--circuit-node-shadow)',
      }}
      className={`absolute top-0 left-0 rounded-[14px] bg-[var(--circuit-node-bg)] border transition-[border-color,opacity] select-none cursor-grab active:cursor-grabbing ${
        isExecuting ? 'circuit-node-executing border-cyan-400/80' : ''
      } ${
        hasError && !selected
          ? 'border-rose-500/65'
          : selected
            ? 'border-cyan-500/70'
            : 'border-[var(--circuit-node-border)] hover:border-[var(--circuit-node-border-hover)]'
      } ${node.disabled ? 'opacity-50 saturate-50' : ''}`}
    >
      {/* 3px Category Accent Cap */}
      <div
        className="h-[3px] w-full rounded-t-[13px]"
        style={{ backgroundColor: catColor }}
      />

      {/* Node Header (exact NODE_HEADER_HEIGHT - 3px = 53px) */}
      <div
        style={{ height: NODE_HEADER_HEIGHT - 3 }}
        className="px-3 flex flex-col justify-center bg-[var(--circuit-node-header)] border-b border-[var(--border-subtle)]"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <div
              className="h-6 w-6 rounded-[7px] flex items-center justify-center shrink-0"
              style={{
                backgroundColor: `${catColor}1F`,
                color: catColor,
              }}
            >
              <Icon className="h-3.5 w-3.5" />
            </div>
            <span className="text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--text-primary)] truncate">
              {node.label || def.shortName}
            </span>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {hasError && (
              <AlertTriangle className="h-3 w-3 text-rose-500" />
            )}
            <span className="relative flex h-2 w-2">
              <span className={`h-2 w-2 rounded-full ${statusDotClass}`} />
            </span>
          </div>
        </div>

        {/* Monospace Live Hyperparameter Readout */}
        <div className="font-mono-num text-[10px] text-[var(--text-muted)] truncate mt-0.5 pl-8">
          {node.disabled ? 'DISABLED · Bypassed' : paramSummary}
        </div>
      </div>

      {/* Port Rail: Exact geometry for sub-pixel SVG alignment */}
      <div
        style={{
          paddingTop: PORT_AREA_TOP_PAD,
          paddingBottom: PORT_AREA_BOTTOM_PAD,
          height: PORT_AREA_TOP_PAD + portRowCount * PORT_ROW_HEIGHT + PORT_AREA_BOTTOM_PAD,
        }}
        className="relative"
      >
        {Array.from({ length: portRowCount }).map((_, idx) => {
          const inp = def.inputs[idx];
          const out = def.outputs[idx];

          const compatInfo =
            inp && wiringDraft ? isPortCompatible(node.id, inp.id) : null;
          const isHoveredTarget =
            inp &&
            wiringDraft?.hoveredTarget?.nodeId === node.id &&
            wiringDraft?.hoveredTarget?.portId === inp.id;

          return (
            <div
              key={idx}
              style={{ height: PORT_ROW_HEIGHT }}
              className="relative flex items-center justify-between px-3"
            >
              {/* Left Input Socket */}
              {inp ? (
                <div
                  className={`flex items-center gap-1.5 min-w-0 max-w-[106px] transition-opacity ${
                    wiringDraft && compatInfo && !compatInfo.compatible ? 'opacity-40' : 'opacity-100'
                  }`}
                  onPointerEnter={() => {
                    if (wiringDraft) onHoverInputPort(node.id, inp);
                  }}
                  onPointerLeave={() => {
                    if (wiringDraft) onHoverInputPort(node.id, null);
                  }}
                  onPointerUp={(e) => {
                    if (wiringDraft) {
                      e.stopPropagation();
                      onCompleteWiring(e, node.id, inp);
                    }
                  }}
                  onPointerDown={(e) => {
                    if (wiringDraft) {
                      e.stopPropagation();
                      onCompleteWiring(e, node.id, inp);
                    }
                  }}
                >
                  {/* Tactile Hardware Pin sitting cleanly on left edge */}
                  <button
                    type="button"
                    data-port-input={`${node.id}:${inp.id}`}
                    aria-label={`Input ${inp.label} (${inp.signalType}) on ${def.shortName}`}
                    title={`${inp.label} · ${inp.signalType}${inp.acceptsFeedback ? ' · Accepts ↻ DMTA Feedback' : ''}${inp.required ? ' (Required)' : ''}`}
                    style={{
                      backgroundColor: getSignalColor(inp.signalType, theme),
                      borderColor: 'var(--circuit-socket-bezel)',
                    }}
                    className={`absolute -left-[6.5px] h-[13px] w-[13px] rounded-full border-2 transition-transform cursor-crosshair z-10 ${
                      wiringDraft && compatInfo?.compatible
                        ? 'circuit-socket-compatible scale-125'
                        : 'hover:scale-125'
                    } ${
                      isHoveredTarget && compatInfo && !compatInfo.compatible
                        ? 'ring-2 ring-rose-500 scale-110'
                        : isHoveredTarget && compatInfo?.wouldBeFeedback
                          ? 'ring-2 ring-fuchsia-400 scale-135'
                          : isHoveredTarget && compatInfo?.compatible
                            ? 'ring-2 ring-emerald-400 scale-135'
                            : ''
                    }`}
                  />
                  <span className="text-[10.5px] font-medium text-[var(--text-secondary)] truncate pl-1 flex items-center gap-0.5">
                    {inp.acceptsFeedback && (
                      <RotateCcw className="h-2.5 w-2.5 text-fuchsia-500 shrink-0" />
                    )}
                    <span className="truncate">{inp.label}</span>
                  </span>
                </div>
              ) : (
                <div />
              )}

              {/* Right Output Socket */}
              {out ? (
                <div
                  className="flex items-center gap-1.5 min-w-0 max-w-[106px] justify-end"
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    onStartWiring(e, node.id, out);
                  }}
                >
                  <span className="text-[10.5px] font-medium text-[var(--text-secondary)] truncate pr-1">
                    {out.label}
                  </span>
                  <button
                    type="button"
                    data-port-output={`${node.id}:${out.id}`}
                    aria-label={`Output ${out.label} (${out.signalType}) on ${def.shortName}`}
                    title={`Drag or click to wire: ${out.label} (${out.signalType})${out.shapeHint ? ` · ${out.shapeHint}` : ''}`}
                    style={{
                      backgroundColor: getSignalColor(out.signalType, theme),
                      borderColor: 'var(--circuit-socket-bezel)',
                    }}
                    className={`absolute -right-[6.5px] h-[13px] w-[13px] rounded-full border-2 transition-transform cursor-crosshair hover:scale-130 z-10 ${
                      wiringDraft?.sourceNodeId === node.id && wiringDraft?.sourcePortId === out.id
                        ? 'scale-135 ring-2 ring-cyan-400'
                        : ''
                    }`}
                  />
                </div>
              ) : (
                <div />
              )}
            </div>
          );
        })}
      </div>

      {/* Live Post-Run Telemetry Pill Footer */}
      {latestTelemetry && (
        <div
          style={{ height: NODE_TELEMETRY_FOOTER_HEIGHT }}
          className="px-2.5 border-t border-[var(--border-subtle)] bg-emerald-500/[0.06] dark:bg-emerald-400/[0.07] rounded-b-[13px] flex items-center justify-between gap-1.5 font-mono-num text-[9.5px]"
        >
          <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300 font-medium truncate">
            <Check className="h-3 w-3 shrink-0" />
            <span className="truncate">{latestTelemetry.summary}</span>
          </span>
          <span className="text-[var(--text-muted)] shrink-0">
            {latestTelemetry.durationMs.toFixed(0)}ms
          </span>
        </div>
      )}
    </div>
  );
}
