import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Copy,
  Info,
  Power,
  RefreshCw,
  Sliders,
  Trash2,
} from 'lucide-react';
import type {
  CircuitEdge,
  CircuitGraphV1,
  CircuitNode,
  ParamSpec,
  ValidationResult,
} from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import {
  BLOCK_MAP,
  getCategoryColor,
  getSignalColor,
  SIGNAL_META,
} from '../../lib/circuitCatalog';
import { CATEGORY_ICONS } from './categoryIcons';

interface InspectorPanelProps {
  circuit: CircuitGraphV1;
  validation: ValidationResult;
  selectedNode: CircuitNode | null;
  selectedEdge: CircuitEdge | null;
  onUpdateNodeParam: (nodeId: string, key: string, value: number | string | boolean) => void;
  onToggleNodeDisabled: (nodeId: string) => void;
  onDuplicateNode: (nodeId: string) => void;
  onDeleteNode: (nodeId: string) => void;
  onDeleteEdge: (edgeId: string) => void;
  onToggleEdgeFeedback?: (edgeId: string) => void;
  onUpdateExecution: (patch: Partial<CircuitGraphV1['execution']>) => void;
  onFocusNode: (nodeId: string) => void;
  theme: ThemeMode;
}

export function InspectorPanel({
  circuit,
  validation,
  selectedNode,
  selectedEdge,
  onUpdateNodeParam,
  onToggleNodeDisabled,
  onDuplicateNode,
  onDeleteNode,
  onDeleteEdge,
  onToggleEdgeFeedback,
  onUpdateExecution,
  onFocusNode,
  theme,
}: InspectorPanelProps) {
  // 1. Selected Node Inspector
  if (selectedNode) {
    const def = BLOCK_MAP[selectedNode.blockType];
    if (!def) return null;
    const catColor = getCategoryColor(def.category, theme);
    const Icon = CATEGORY_ICONS[def.category];
    const nodeDiags = validation.diagnostics.filter((d) => d.nodeId === selectedNode.id);

    const renderParamControl = (spec: ParamSpec) => {
      const value = selectedNode.params[spec.key] ?? spec.default;

      if (spec.type === 'boolean') {
        const checked = Boolean(value);
        return (
          <div key={spec.key} className="flex items-center justify-between gap-2 py-1">
            <span className="text-[11.5px] font-medium text-[var(--text-secondary)]">
              {spec.label}
            </span>
            <button
              type="button"
              role="switch"
              aria-checked={checked}
              onClick={() => onUpdateNodeParam(selectedNode.id, spec.key, !checked)}
              className={`relative h-5 w-9 rounded-full transition-colors cursor-pointer shrink-0 ${
                checked ? 'bg-cyan-500' : 'bg-slate-400/40 dark:bg-slate-700'
              }`}
            >
              <span
                className={`block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                  checked ? 'translate-x-[18px]' : 'translate-x-[2px]'
                }`}
              />
            </button>
          </div>
        );
      }

      if (spec.type === 'number') {
        const num = Number(value);
        const min = spec.min ?? 0;
        const max = spec.max ?? 100;
        const step = spec.step ?? 1;
        return (
          <div key={spec.key} className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-[11.5px]">
              <label className="font-medium text-[var(--text-secondary)]">{spec.label}</label>
              <span className="font-mono-num text-[11px] font-bold text-[var(--text-primary)]">
                {Number.isInteger(step) ? num : num.toFixed(step < 0.05 ? 3 : 2)}
                {spec.unit ? ` ${spec.unit}` : ''}
              </span>
            </div>
            <input
              type="range"
              data-param-key={spec.key}
              min={min}
              max={max}
              step={step}
              value={num}
              onChange={(ev) => onUpdateNodeParam(selectedNode.id, spec.key, Number(ev.target.value))}
              className="w-full apple-range"
            />
          </div>
        );
      }

      if (spec.type === 'select' && spec.options) {
        const isCompact = spec.options.length <= 3 && spec.options.every((o) => String(o).length <= 12);
        return (
          <div key={spec.key} className="space-y-1">
            <label className="block text-[11.5px] font-medium text-[var(--text-secondary)]">
              {spec.label}
            </label>
            {isCompact ? (
              <div className="grid grid-cols-3 gap-1">
                {spec.options.map((opt) => {
                  const active = String(opt) === String(value);
                  return (
                    <button
                      key={String(opt)}
                      type="button"
                      onClick={() =>
                        onUpdateNodeParam(
                          selectedNode.id,
                          spec.key,
                          typeof spec.default === 'number' ? Number(opt) : String(opt),
                        )
                      }
                      className={`py-1 px-1.5 rounded-lg text-[10.5px] font-mono-num font-semibold border transition-all truncate cursor-pointer ${
                        active
                          ? 'bg-cyan-500/18 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                          : 'border-[var(--circuit-node-border)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      {String(opt)}
                    </button>
                  );
                })}
              </div>
            ) : (
              <select
                data-param-key={spec.key}
                value={String(value)}
                onChange={(ev) =>
                  onUpdateNodeParam(
                    selectedNode.id,
                    spec.key,
                    typeof spec.default === 'number' ? Number(ev.target.value) : ev.target.value,
                  )
                }
                className="w-full h-8 px-2.5 rounded-lg text-[11.5px] font-mono-num bg-[var(--bg-elevated)] border border-[var(--circuit-node-border)] text-[var(--text-primary)] outline-none focus:border-cyan-500"
              >
                {spec.options.map((opt) => (
                  <option key={String(opt)} value={String(opt)}>
                    {String(opt)}
                  </option>
                ))}
              </select>
            )}
          </div>
        );
      }

      return (
        <div key={spec.key} className="space-y-1">
          <label className="block text-[11.5px] font-medium text-[var(--text-secondary)]">
            {spec.label}
          </label>
          <input
            type="text"
            data-param-key={spec.key}
            value={String(value)}
            onChange={(ev) => onUpdateNodeParam(selectedNode.id, spec.key, ev.target.value)}
            className="w-full h-8 px-2.5 rounded-lg text-[11px] font-mono-num bg-[var(--bg-elevated)] border border-[var(--circuit-node-border)] text-[var(--text-primary)] outline-none focus:border-cyan-500"
          />
        </div>
      );
    };

    return (
      <div className="p-3.5 space-y-4">
        {/* Block Header */}
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <div
                className="h-8 w-8 rounded-xl flex items-center justify-center shrink-0"
                style={{
                  backgroundColor: `${catColor}1F`,
                  color: catColor,
                  border: `1px solid ${catColor}3A`,
                }}
              >
                <Icon className="h-4 w-4" />
              </div>
              <div className="min-w-0">
                <h3 className="text-[13.5px] font-bold tracking-tight text-[var(--text-primary)] truncate">
                  {def.name}
                </h3>
                <div className="font-mono-num text-[10px] text-[var(--text-muted)] truncate">
                  id: {selectedNode.id} · {def.executionTier}
                </div>
              </div>
            </div>
            <span
              className="font-mono-num text-[10px] font-bold px-2 py-0.5 rounded-md shrink-0"
              style={{ backgroundColor: `${catColor}18`, color: catColor }}
            >
              {def.badge}
            </span>
          </div>

          <div className="rounded-lg bg-[var(--bg-elevated)]/85 border border-[var(--circuit-node-border)] px-2.5 py-1.5 font-mono-num text-[10px] text-cyan-700 dark:text-cyan-300 break-all">
            {def.pythonSymbol}
          </div>

          <p className="text-[11.5px] text-[var(--text-secondary)] leading-relaxed">{def.summary}</p>
        </div>

        {/* Node-Specific Validation Diagnostics */}
        {nodeDiags.length > 0 && (
          <div className="space-y-1.5">
            {nodeDiags.map((d, i) => (
              <div
                key={`${d.code}-${i}`}
                className={`rounded-xl px-2.5 py-2 text-[11px] border ${
                  d.severity === 'error'
                    ? 'bg-rose-500/10 border-rose-500/35 text-rose-700 dark:text-rose-200'
                    : 'bg-amber-500/10 border-amber-500/35 text-amber-800 dark:text-amber-200'
                }`}
              >
                <div className="font-semibold">{d.message}</div>
                {d.remediation && <div className="opacity-85 mt-0.5">{d.remediation}</div>}
              </div>
            ))}
          </div>
        )}

        {/* Hyperparameters */}
        <div className="space-y-2.5 pt-2 border-t border-[var(--circuit-node-border)]">
          <div className="flex items-center gap-1.5 text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
            <Sliders className="h-3 w-3" /> Real GenEuclidean Parameters
          </div>
          <div className="space-y-3">{def.params.map(renderParamControl)}</div>
        </div>

        {/* Typed I/O Signal Contracts */}
        <div className="space-y-2 pt-2 border-t border-[var(--circuit-node-border)]">
          <div className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
            Typed Port Contracts
          </div>
          <div className="space-y-1.5">
            {def.inputs.map((p) => {
              const c = getSignalColor(p.signalType, theme);
              return (
                <div
                  key={`in-${p.id}`}
                  className="flex items-center justify-between gap-2 text-[10.5px] px-2 py-1 rounded-lg bg-[var(--bg-elevated)]/70"
                >
                  <span className="flex items-center gap-1.5 text-[var(--text-primary)] font-medium">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: c }} />
                    IN · {p.label}
                    {p.acceptsFeedback && (
                      <RefreshCw className="h-2.5 w-2.5 text-pink-500" />
                    )}
                  </span>
                  <span className="font-mono-num text-[9.5px]" style={{ color: c }}>
                    {SIGNAL_META[p.signalType].shortLabel}
                  </span>
                </div>
              );
            })}
            {def.outputs.map((p) => {
              const c = getSignalColor(p.signalType, theme);
              return (
                <div
                  key={`out-${p.id}`}
                  className="flex items-center justify-between gap-2 text-[10.5px] px-2 py-1 rounded-lg bg-[var(--bg-elevated)]/70"
                >
                  <span className="flex items-center gap-1.5 text-[var(--text-primary)] font-medium">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: c }} />
                    OUT · {p.label}
                  </span>
                  <span className="font-mono-num text-[9.5px] text-[var(--text-muted)] truncate max-w-[130px]">
                    {p.shapeHint ?? SIGNAL_META[p.signalType].shortLabel}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Quick Node Actions */}
        <div className="grid grid-cols-3 gap-1.5 pt-2 border-t border-[var(--circuit-node-border)]">
          <button
            type="button"
            onClick={() => onDuplicateNode(selectedNode.id)}
            className="py-1.5 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 flex items-center justify-center gap-1 text-[var(--text-primary)] cursor-pointer"
          >
            <Copy className="h-3 w-3" /> Copy
          </button>
          <button
            type="button"
            onClick={() => onToggleNodeDisabled(selectedNode.id)}
            className="py-1.5 rounded-lg text-[11px] font-semibold border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-amber-500/45 flex items-center justify-center gap-1 text-[var(--text-primary)] cursor-pointer"
          >
            <Power className="h-3 w-3" /> {selectedNode.disabled ? 'Enable' : 'Bypass'}
          </button>
          <button
            type="button"
            onClick={() => onDeleteNode(selectedNode.id)}
            className="py-1.5 rounded-lg text-[11px] font-semibold border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 flex items-center justify-center gap-1 text-rose-600 dark:text-rose-300 cursor-pointer"
          >
            <Trash2 className="h-3 w-3" /> Delete
          </button>
        </div>
      </div>
    );
  }

  // 2. Selected Edge Inspector
  if (selectedEdge) {
    const srcNode = circuit.nodes.find((n) => n.id === selectedEdge.source);
    const dstNode = circuit.nodes.find((n) => n.id === selectedEdge.target);
    const srcDef = srcNode ? BLOCK_MAP[srcNode.blockType] : undefined;
    const dstDef = dstNode ? BLOCK_MAP[dstNode.blockType] : undefined;
    const sigColor = getSignalColor(selectedEdge.signalType, theme);
    const sigMeta = SIGNAL_META[selectedEdge.signalType];

    const dstPort = dstDef?.inputs.find((p) => p.id === selectedEdge.targetPort);

    return (
      <div className="p-3.5 space-y-4">
        <div className="space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-bold text-[var(--text-primary)]">Signal Wire</span>
            <span
              className="font-mono-num text-[10px] font-bold px-2 py-0.5 rounded-full"
              style={{ backgroundColor: `${sigColor}20`, color: sigColor }}
            >
              {selectedEdge.kind === 'feedback' ? '↻ DMTA Feedback' : sigMeta.label}
            </span>
          </div>
          <div className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--circuit-node-border)] p-3 space-y-1 font-mono-num text-[11px]">
            <div className="text-[var(--text-primary)] font-semibold">
              {srcDef?.shortName ?? selectedEdge.source}.{selectedEdge.sourcePort}
            </div>
            <div className="text-[var(--text-muted)]">↓ {selectedEdge.signalType}</div>
            <div className="text-[var(--text-primary)] font-semibold">
              {dstDef?.shortName ?? selectedEdge.target}.{selectedEdge.targetPort}
            </div>
          </div>
          <p className="text-[11px] text-[var(--text-secondary)]">
            <strong className="text-[var(--text-primary)]">Python Contract:</strong>{' '}
            <code className="font-mono-num text-[10.5px]">{sigMeta.pythonContract}</code>
          </p>
        </div>

        {dstPort?.acceptsFeedback && onToggleEdgeFeedback && (
          <button
            type="button"
            onClick={() => onToggleEdgeFeedback(selectedEdge.id)}
            className="w-full py-2 rounded-xl text-xs font-semibold border border-fuchsia-500/35 bg-fuchsia-500/10 hover:bg-fuchsia-500/18 text-fuchsia-700 dark:text-fuchsia-300 flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            {selectedEdge.kind === 'feedback' ? 'Convert to Forward Wire' : 'Convert to ↻ DMTA Feedback Loop'}
          </button>
        )}

        <button
          type="button"
          onClick={() => onDeleteEdge(selectedEdge.id)}
          className="w-full py-2 rounded-xl text-xs font-semibold border border-rose-500/35 bg-rose-500/10 hover:bg-rose-500/20 text-rose-600 dark:text-rose-300 flex items-center justify-center gap-1.5 cursor-pointer"
        >
          <Trash2 className="h-3.5 w-3.5" /> Disconnect Wire
        </button>
      </div>
    );
  }

  // 3. Global Circuit Overview & Diagnostics
  return (
    <div className="p-3.5 space-y-4">
      <div className="space-y-1">
        <h3 className="text-[13.5px] font-bold tracking-tight text-[var(--text-primary)]">
          {circuit.name}
        </h3>
        <p className="text-[11.5px] text-[var(--text-secondary)] leading-relaxed">
          {circuit.description || 'Click any block on the canvas to inspect and tune its real GenEuclidean hyperparameters.'}
        </p>
      </div>

      {/* Global Execution Settings */}
      <div className="space-y-3 pt-2 border-t border-[var(--circuit-node-border)]">
        <div className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
          Circuit Execution Controls
        </div>

        <div className="grid grid-cols-2 gap-2.5">
          <div className="space-y-1">
            <label className="text-[11px] font-medium text-[var(--text-secondary)]">Random Seed</label>
            <input
              type="number"
              value={circuit.execution.seed}
              onChange={(ev) => onUpdateExecution({ seed: Number(ev.target.value) || 42 })}
              className="w-full h-8 px-2.5 rounded-lg text-xs font-mono-num bg-[var(--bg-elevated)] border border-[var(--circuit-node-border)] text-[var(--text-primary)]"
            />
          </div>
          <div className="space-y-1">
            <label className="text-[11px] font-medium text-[var(--text-secondary)]">
              DMTA Rounds
            </label>
            <input
              type="number"
              min={1}
              max={10}
              value={circuit.execution.dmtaRounds}
              onChange={(ev) =>
                onUpdateExecution({
                  dmtaRounds: Math.max(1, Math.min(10, Number(ev.target.value) || 1)),
                })
              }
              className="w-full h-8 px-2.5 rounded-lg text-xs font-mono-num bg-[var(--bg-elevated)] border border-[var(--circuit-node-border)] text-[var(--text-primary)]"
            />
          </div>
        </div>
      </div>

      {/* Validation Diagnostics */}
      <div className="space-y-2 pt-2 border-t border-[var(--circuit-node-border)]">
        <div className="flex items-center justify-between">
          <span className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
            Topology &amp; Signal Checks
          </span>
          <span
            className={`font-mono-num text-[10px] font-bold px-2 py-0.5 rounded-full ${
              validation.valid
                ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
                : 'bg-rose-500/15 text-rose-600 dark:text-rose-300'
            }`}
          >
            {validation.valid ? 'PASS' : 'BLOCKED'}
          </span>
        </div>

        {validation.diagnostics.length === 0 ? (
          <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/8 px-3 py-2.5 flex items-center gap-2 text-xs text-emerald-700 dark:text-emerald-300">
            <CheckCircle2 className="h-4 w-4 shrink-0" />
            <span>
              All {circuit.nodes.length} blocks &amp; {circuit.edges.length} signal wires verified.
              {validation.hasFeedbackLoop ? ' Closed-loop DMTA feedback active.' : ''}
            </span>
          </div>
        ) : (
          <div className="space-y-1.5">
            {validation.diagnostics.map((d, i) => (
              <button
                key={`${d.code}-${i}`}
                type="button"
                onClick={() => d.nodeId && onFocusNode(d.nodeId)}
                className={`w-full text-left rounded-xl px-2.5 py-2 text-[11px] border transition-colors ${
                  d.nodeId ? 'cursor-pointer hover:brightness-105' : 'cursor-default'
                } ${
                  d.severity === 'error'
                    ? 'bg-rose-500/10 border-rose-500/35 text-rose-700 dark:text-rose-200'
                    : d.severity === 'warning'
                      ? 'bg-amber-500/10 border-amber-500/35 text-amber-800 dark:text-amber-200'
                      : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-800 dark:text-cyan-200'
                }`}
              >
                <div className="flex items-start gap-1.5">
                  {d.severity === 'error' ? (
                    <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                  ) : d.severity === 'warning' ? (
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                  ) : (
                    <Info className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                  )}
                  <div>
                    <div className="font-semibold">{d.message}</div>
                    {d.remediation && <div className="opacity-85 mt-0.5">{d.remediation}</div>}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Quick Keyboard Shortcuts Cheat Sheet */}
      <div className="pt-2 border-t border-[var(--circuit-node-border)] space-y-1.5">
        <div className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
          Keyboard Shortcuts
        </div>
        <div className="grid grid-cols-2 gap-1.5 text-[10.5px] text-[var(--text-secondary)]">
          {[
            ['⌘Enter', 'Run Circuit'],
            ['⌘K', 'Search Blocks'],
            ['0 / L', 'Fit / Auto-Layout'],
            ['⌘Z / ⇧⌘Z', 'Undo / Redo'],
            ['⌘D / ⌫', 'Duplicate / Delete'],
            ['Space+Drag', 'Pan Canvas'],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-1 px-2 py-1 rounded-md bg-[var(--bg-elevated)]/65">
              <span className="font-mono-num text-[9.5px] font-bold text-[var(--text-primary)]">{k}</span>
              <span className="text-[10px] text-[var(--text-muted)]">{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
