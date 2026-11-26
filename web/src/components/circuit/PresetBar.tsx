import {
  AlertCircle,
  Atom,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Upload,
} from 'lucide-react';
import { useRef } from 'react';
import type {
  CircuitPreset,
  ExecutionEngineMode,
  LocalEngineHealth,
  ValidationResult,
} from '../../types/circuit';

interface PresetBarProps {
  presets: CircuitPreset[];
  activePresetId: string | null;
  onSelectPreset: (presetId: string) => void;
  validation: ValidationResult;
  nodeCount: number;
  edgeCount: number;
  dmtaRounds?: number;
  engineMode: ExecutionEngineMode;
  onChangeEngineMode: (mode: ExecutionEngineMode) => void;
  localHealth: LocalEngineHealth;
  onRecheckLocal?: () => void;
  onImportJson: (text: string) => void;
  onExportJson: () => void;
  onRunCircuit: () => void;
  onFocusFirstIssue: () => void;
  isRunning: boolean;
}

export function PresetBar({
  presets,
  activePresetId,
  onSelectPreset,
  validation,
  nodeCount,
  edgeCount,
  dmtaRounds = 5,
  engineMode,
  onChangeEngineMode,
  localHealth,
  onRecheckLocal,
  onImportJson,
  onExportJson,
  onRunCircuit,
  onFocusFirstIssue,
  isRunning,
}: PresetBarProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const errorCount = validation.diagnostics.filter((d) => d.severity === 'error').length;
  const warningCount = validation.diagnostics.filter((d) => d.severity === 'warning').length;

  return (
    <div className="h-12 shrink-0 circuit-glass border-b border-[var(--circuit-node-border)] px-3 flex items-center justify-between gap-2 z-30">
      {/* Left: 7 Flagship One-Click Preset Pills (Apple Segmented Strip) */}
      <div className="flex items-center gap-1.5 min-w-0 flex-1 overflow-hidden">
        <span className="hidden 2xl:inline text-[10px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold shrink-0">
          Presets
        </span>
        <div className="flex items-center gap-0.5 p-0.5 rounded-xl border border-[var(--circuit-node-border)] bg-[var(--bg-elevated)]/70 overflow-x-auto no-scrollbar min-w-0">
          {presets.map((preset) => {
            const active = activePresetId === preset.id;
            return (
              <button
                key={preset.id}
                type="button"
                data-preset-id={preset.id}
                title={`${preset.name} — ${preset.tagline}`}
                onClick={() => onSelectPreset(preset.id)}
                className={`px-2 py-1 rounded-[9px] text-[11px] font-medium whitespace-nowrap transition-all flex items-center gap-1 border cursor-pointer shrink-0 ${
                  active
                    ? 'bg-cyan-500/16 text-cyan-700 dark:text-cyan-200 border-cyan-500/40 shadow-xs font-semibold'
                    : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-slate-500/10'
                }`}
              >
                <span>{preset.shortLabel}</span>
                {preset.id === 'al_dmta_closed_loop' && (
                  <RefreshCw className="h-2.5 w-2.5 text-pink-500 dark:text-pink-400 shrink-0" />
                )}
                {preset.id === 'nanofold3_cofold_audit' && (
                  <Atom className="h-2.5 w-2.5 text-teal-500 dark:text-teal-300 shrink-0" />
                )}
                {preset.id === 'tourette_scz_champion' && (
                  <Sparkles className="h-2.5 w-2.5 text-emerald-500 dark:text-emerald-300 shrink-0" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Center: Compact Live Circuit Status Capsule */}
      <button
        type="button"
        onClick={onFocusFirstIssue}
        title="Inspect circuit topology & validation diagnostics"
        className={`hidden xl:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono-num border transition-colors shrink-0 cursor-pointer ${
          errorCount > 0
            ? 'bg-rose-500/12 border-rose-500/35 text-rose-600 dark:text-rose-300'
            : warningCount > 0
              ? 'bg-amber-500/12 border-amber-500/35 text-amber-700 dark:text-amber-300'
              : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-300'
        }`}
      >
        {errorCount > 0 ? (
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
        )}
        <span className="font-semibold">
          {errorCount > 0
            ? `${errorCount} Issue${errorCount > 1 ? 's' : ''}`
            : validation.hasFeedbackLoop
              ? `Valid Loop (${dmtaRounds}r)`
              : nodeCount === 0
                ? 'Blank'
                : 'Valid DAG'}
        </span>
        <span className="opacity-65">·</span>
        <span>
          {nodeCount}B / {edgeCount}W
        </span>
      </button>

      {/* Right: Engine Toggle + JSON Import/Export + Run Circuit */}
      <div className="flex items-center gap-1.5 shrink-0">
        {/* Segmented Dual-Mode Engine Toggle */}
        <div
          className="flex items-center p-0.5 rounded-lg border border-[var(--circuit-node-border)] bg-[var(--bg-elevated)]/80"
          title={
            localHealth.reachable
              ? `Local Python Engine online (v${localHealth.version}, ${localHealth.catalogCount} blocks)`
              : 'Browser Sim runs zero-backend on GitHub Pages. To execute real local Python: geneuclidean studio --port 8765'
          }
        >
          <button
            type="button"
            onClick={() => onChangeEngineMode('browser_sim')}
            className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all cursor-pointer ${
              engineMode === 'browser_sim'
                ? 'bg-[var(--bg-surface)] text-[var(--text-primary)] shadow-xs font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
          >
            Browser Sim
          </button>
          <button
            type="button"
            onClick={() => {
              onChangeEngineMode('local_python');
              onRecheckLocal?.();
            }}
            className={`px-2 py-1 rounded-md text-[11px] font-medium transition-all flex items-center gap-1.5 cursor-pointer ${
              engineMode === 'local_python'
                ? 'bg-[var(--bg-surface)] text-[var(--text-primary)] shadow-xs font-semibold'
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                localHealth.checking
                  ? 'bg-cyan-400 animate-pulse'
                  : localHealth.reachable
                    ? 'bg-emerald-400'
                    : 'bg-amber-400'
              }`}
            />
            <Server className="h-3 w-3 hidden sm:inline" />
            <span>Local :8765</span>
          </button>
        </div>

        {/* Hidden file input for JSON import */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(ev) => {
            const file = ev.target.files?.[0];
            if (!file) return;
            file.text().then((txt) => onImportJson(txt));
            ev.target.value = '';
          }}
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Import CircuitGraphV1 .json"
          aria-label="Import JSON"
          className="h-8 w-8 rounded-lg border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
        >
          <Upload className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          onClick={onExportJson}
          title="Export CircuitGraphV1 .json"
          aria-label="Export JSON"
          className="h-8 w-8 rounded-lg border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
        >
          <Download className="h-3.5 w-3.5" />
        </button>

        {/* Primary Apple Precision CTA */}
        <button
          type="button"
          data-run-circuit="true"
          disabled={isRunning || nodeCount === 0}
          onClick={onRunCircuit}
          title="Execute circuit (⌘Enter)"
          className={`h-8 px-3.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
            isRunning || nodeCount === 0
              ? 'bg-slate-500/20 text-[var(--text-muted)] cursor-not-allowed'
              : 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-md shadow-cyan-500/20 hover:brightness-110 active:scale-[0.98] cursor-pointer'
          }`}
        >
          {isRunning ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5 fill-current" />
          )}
          <span>{isRunning ? 'Running…' : 'Run Circuit'}</span>
        </button>
      </div>
    </div>
  );
}
