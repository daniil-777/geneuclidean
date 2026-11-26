import { AlertTriangle, Check, RotateCcw } from 'lucide-react';
import type React from 'react';
import type {
  CircuitNode as CircuitNodeType,
  ExecutionStepTelemetry,
  NodeRuntimeStatus,
  PortSpec,
  WiringDraft,
} from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import {
  BLOCK_MAP,
  getCategoryColor,
  getNodeParamSummary,
  getSignalColor,
} from '../../lib/circuitCatalog';
import {
  NODE_FOOTER_H,
  NODE_HEADER_H,
  NODE_PORT_PAD_BOTTOM,
  NODE_PORT_PAD_TOP,
  NODE_PORT_ROW_H,
  NODE_SUBTITLE_H,
  NODE_TOP_CAP,
  NODE_WIDTH,
} from '../../lib/circuitEngine';
import { CATEGORY_ICONS } from './categoryIcons';

interface CircuitNodeProps {
  node: CircuitNodeType;
  theme: ThemeMode;
  selected: boolean;
  hasError: boolean;
  runtimeStatus: NodeRuntimeStatus;
  latestTelemetry?: ExecutionStepTelemetry;
  wiringDraft: WiringDraft | null;
  checkInputCompat: (
    targetNodeId: string,
    targetPortId: string,
  ) => { compatible: boolean; kind: 'forward' | 'feedback'; hint?: string };
  onNodePointerDown: (e: React.PointerEvent, nodeId: string) => void;
  onStartWiring: (e: React.PointerEvent, nodeId: string, port: PortSpec) => void;
  onCompleteWiring: (e: React.PointerEvent, nodeId: string, port: PortSpec) => void;
  onHoverInputPort: (nodeId: string, port: PortSpec | null) => void;
}

export function CircuitNode({
  node,
  theme,
  selected,
  hasError,
  runtimeStatus,
  latestTelemetry,
  wiringDraft,
  checkInputCompat,
  onNodePointerDown,
  onStartWiring,
  onCompleteWiring,
  onHoverInputPort,
}: CircuitNodeProps) {
  const def = BLOCK_MAP[node.blockType];
  if (!def) return null;

  const Icon = CATEGORY_ICONS[def.category];
  const catColor = getCategoryColor(def.category, theme);
  const subtitle = getNodeParamSummary(node);
  const portRows = Math.max(1, def.inputs.length, def.outputs.length);
  const isRunning = runtimeStatus === 'running';

  const ledClass =
    runtimeStatus === 'running'
      ? 'bg-cyan-400 animate-ping'
      : runtimeStatus === 'ok'
        ? 'bg-emerald-500'
        : runtimeStatus === 'simulated_fallback'
          ? 'bg-amber-400'
          : runtimeStatus === 'error' || hasError
            ? 'bg-rose-500'
            : 'bg-slate-400/50';

  return (
    <div
      data-circuit-node={node.id}
      data-block-type={node.blockType}
      onPointerDown={(e) => onNodePointerDown(e, node.id)}
      style={{
        transform: `translate3d(${node.position.x}px, ${node.position.y}px, 0)`,
        width: NODE_WIDTH,
      }}
      className={`absolute top-0 left-0 rounded-[14px] circuit-node-chassis select-none cursor-grab active:cursor-grabbing ${
        selected ? 'circuit-node-selected z-20' : 'z-10'
      } ${isRunning ? 'circuit-node-executing' : ''} ${
        hasError && !selected ? '!border-rose-500/70' : ''
      } ${node.disabled ? 'opacity-50 saturate-50' : ''}`}
    >
      {/* 3px Category Accent Cap */}
      <div
        style={{ height: NODE_TOP_CAP, backgroundColor: catColor }}
        className="w-full rounded-t-[13px]"
      />

      {/* Node Header (exact 42px) */}
      <div
        style={{ height: NODE_HEADER_H }}
        className="px-3 flex items-center justify-between gap-2 bg-[var(--circuit-node-header)]"
      >
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="h-6 w-6 rounded-[7px] flex items-center justify-center shrink-0"
            style={{ backgroundColor: `${catColor}20`, color: catColor }}
          >
            <Icon className="h-3.5 w-3.5" />
          </div>
          <span className="text-[12.5px] font-semibold tracking-[-0.01em] text-[var(--text-primary)] truncate">
            {node.label || def.shortName}
          </span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {hasError && <AlertTriangle className="h-3 w-3 text-rose-500" />}
          <span className={`h-2 w-2 rounded-full ${ledClass}`} />
        </div>
      </div>

      {/* Monospace Live Hyperparameter Strip (exact 24px) */}
      <div
        style={{ height: NODE_SUBTITLE_H }}
        className="px-3 flex items-center justify-between gap-2 bg-[var(--circuit-node-header)] border-b border-[var(--border-subtle)]"
      >
        <span className="font-mono-num text-[10px] text-[var(--text-muted)] truncate">
          {node.disabled ? 'BYPASSED' : subtitle}
        </span>
        <span
          className="font-mono-num text-[9px] font-semibold px-1.5 py-0.2 rounded shrink-0"
          style={{ backgroundColor: `${catColor}18`, color: catColor }}
        >
          {def.badge}
        </span>
      </div>

      {/* Hardware-Tactile Port Rail (exact deterministic heights) */}
      <div
        style={{
          paddingTop: NODE_PORT_PAD_TOP,
          paddingBottom: NODE_PORT_PAD_BOTTOM,
          height: NODE_PORT_PAD_TOP + portRows * NODE_PORT_ROW_H + NODE_PORT_PAD_BOTTOM,
        }}
        className="relative"
      >
        {Array.from({ length: portRows }).map((_, idx) => {
          const inp = def.inputs[idx];
          const out = def.outputs[idx];
          const compat = inp && wiringDraft ? checkInputCompat(node.id, inp.id) : null;
          const isHovered =
            inp &&
            wiringDraft?.hoveredTarget?.nodeId === node.id &&
            wiringDraft?.hoveredTarget?.portId === inp.id;

          return (
            <div
              key={idx}
              style={{ height: NODE_PORT_ROW_H }}
              className="relative flex items-center justify-between px-3"
            >
              {/* Input Socket */}
              {inp ? (
                <div
                  className={`flex items-center gap-1 min-w-0 max-w-[104px] transition-opacity ${
                    wiringDraft && compat && !compat.compatible ? 'opacity-40' : 'opacity-100'
                  }`}
                  onPointerEnter={() => wiringDraft && onHoverInputPort(node.id, inp)}
                  onPointerLeave={() => wiringDraft && onHoverInputPort(node.id, null)}
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
                  <button
                    type="button"
                    data-port-input={`${node.id}:${inp.id}`}
                    aria-label={`Input ${inp.label} (${inp.signalType}) on ${def.shortName}`}
                    title={`${inp.label} (${inp.signalType})${inp.acceptsFeedback ? ' · Accepts ↻ DMTA Feedback' : ''}`}
                    style={{
                      backgroundColor: getSignalColor(inp.signalType, theme),
                      borderColor: 'var(--circuit-socket-bezel)',
                    }}
                    className={`absolute -left-[6.5px] h-[13px] w-[13px] rounded-full border-2 transition-transform cursor-crosshair z-20 ${
                      wiringDraft && compat?.compatible
                        ? 'circuit-pin-compatible scale-125'
                        : 'hover:scale-125'
                    } ${
                      isHovered && compat && !compat.compatible
                        ? 'ring-2 ring-rose-500 scale-115'
                        : isHovered && compat?.kind === 'feedback'
                          ? 'ring-2 ring-fuchsia-400 scale-135'
                          : isHovered && compat?.compatible
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

              {/* Output Socket */}
              {out ? (
                <div
                  className="flex items-center gap-1 min-w-0 max-w-[104px] justify-end"
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
                    title={`Drag or click to wire: ${out.label} (${out.signalType})`}
                    style={{
                      backgroundColor: getSignalColor(out.signalType, theme),
                      borderColor: 'var(--circuit-socket-bezel)',
                    }}
                    className={`absolute -right-[6.5px] h-[13px] w-[13px] rounded-full border-2 transition-transform cursor-crosshair hover:scale-125 z-20 ${
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

      {/* Live Post-Run Telemetry Strip (exact 26px) */}
      {latestTelemetry && (
        <div
          style={{ height: NODE_FOOTER_H }}
          className="px-2.5 border-t border-[var(--border-subtle)] bg-emerald-500/[0.08] dark:bg-emerald-400/[0.08] rounded-b-[13px] flex items-center justify-between gap-1.5 font-mono-num text-[9.5px]"
        >
          <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300 font-semibold truncate">
            <Check className="h-3 w-3 shrink-0" />
            <span className="truncate">{latestTelemetry.summary}</span>
          </span>
          <span className="text-[var(--text-muted)] shrink-0">{latestTelemetry.durationMs}ms</span>
        </div>
      )}
    </div>
  );
}
