import {
  Activity,
  ArrowUpRight,
  Check,
  CheckCircle2,
  Copy,
  Play,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { RunCircuitResponse } from '../../types/circuit';
import type { MoleculeIndexEntry, ThemeMode } from '../../types/studio';
import { BLOCK_MAP } from '../../lib/circuitCatalog';
import { copyToClipboard } from '../../lib/data';

interface ExecutionTelemetryProps {
  runResult: RunCircuitResponse | null;
  isRunning: boolean;
  moleculesIndex: MoleculeIndexEntry[];
  theme: ThemeMode;
  onRunCircuit: () => void;
  onFocusNode: (nodeId: string) => void;
  onInspectIn3D: (target: string) => void;
}

export function ExecutionTelemetry({
  runResult,
  isRunning,
  moleculesIndex,
  theme,
  onRunCircuit,
  onFocusNode,
  onInspectIn3D,
}: ExecutionTelemetryProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = async (key: string, text: string) => {
    if (await copyToClipboard(text)) {
      setCopiedId(key);
      setTimeout(() => setCopiedId(null), 1400);
    }
  };

  if (!runResult) {
    return (
      <div className="p-5 text-center space-y-3.5">
        <div className="mx-auto h-11 w-11 rounded-2xl bg-cyan-500/12 border border-cyan-500/25 flex items-center justify-center text-cyan-500">
          <Activity className="h-5 w-5" />
        </div>
        <div className="space-y-1">
          <h4 className="text-xs font-bold text-[var(--text-primary)]">
            No Execution Telemetry Yet
          </h4>
          <p className="text-[11.5px] text-[var(--text-secondary)] leading-relaxed">
            Click <strong>Run Circuit</strong> (or press <code className="font-mono-num">⌘Enter</code>)
            to watch typed signals propagate across the circuit and inspect verified GenEuclidean
            metrics, DMTA curves, and candidate leads.
          </p>
        </div>
        <button
          type="button"
          disabled={isRunning}
          onClick={onRunCircuit}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-md shadow-cyan-500/20 hover:brightness-110 cursor-pointer"
        >
          <Play className="h-3.5 w-3.5 fill-current" />
          Run Circuit Now
        </button>
      </div>
    );
  }

  const molSvgMap = new Map(moleculesIndex.map((m) => [m.canonical_smiles, m]));
  // Deduplicate steps to latest round per node for crisp timeline display
  const latestSteps = runResult.steps.filter((s) => s.round === runResult.roundsCompleted);
  const axisTickColor = theme === 'dark' ? '#94A3B8' : '#64748B';
  const gridStroke = theme === 'dark' ? 'rgba(148,163,184,0.14)' : 'rgba(15,23,42,0.08)';

  return (
    <div className="p-3.5 space-y-4">
      {/* Run Summary Header */}
      <div className="rounded-xl border border-[var(--circuit-node-border)] bg-[var(--bg-elevated)]/75 p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className="inline-flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-300">
            <CheckCircle2 className="h-3.5 w-3.5" />
            {runResult.engineMode === 'local_python' ? 'Local Python Engine' : 'Browser Simulation'}
          </span>
          <span className="font-mono-num text-[10px] font-semibold px-2 py-0.5 rounded-full bg-cyan-500/12 text-cyan-700 dark:text-cyan-300">
            {runResult.roundsCompleted}r · {runResult.totalDurationMs} ms
          </span>
        </div>

        <div className="flex items-center justify-between gap-2 text-[10px] font-mono-num text-[var(--text-muted)]">
          <button
            type="button"
            onClick={() => handleCopy('sha', runResult.provenanceSha256)}
            className="inline-flex items-center gap-1 hover:text-[var(--text-primary)] cursor-pointer"
            title="Copy full SHA-256 provenance digest"
          >
            <span>SHA-256: {runResult.provenanceSha256.slice(0, 12)}…</span>
            {copiedId === 'sha' ? (
              <Check className="h-2.5 w-2.5 text-emerald-500" />
            ) : (
              <Copy className="h-2.5 w-2.5" />
            )}
          </button>
          <span className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-semibold">
            <ShieldCheck className="h-3 w-3" /> FDA/EMA {runResult.provenanceLedger.fdaEmaPrinciples}
          </span>
        </div>
      </div>

      {/* Interactive Recharts DMTA Convergence Chart (when multi-round loop ran) */}
      {runResult.dmtaTrace.length > 1 && (
        <div className="rounded-xl border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold text-[var(--text-primary)]">
              DMTA Active Learning Convergence
            </span>
            <span className="font-mono-num text-[10px] text-pink-600 dark:text-pink-400 font-semibold">
              {runResult.dmtaTrace.length} Rounds
            </span>
          </div>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={runResult.dmtaTrace} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid stroke={gridStroke} strokeDasharray="3 3" />
                <XAxis
                  dataKey="round"
                  tick={{ fontSize: 10, fill: axisTickColor }}
                  tickFormatter={(v) => `R${v}`}
                />
                <YAxis
                  yAxisId="left"
                  domain={[7.5, 10.2]}
                  tick={{ fontSize: 10, fill: axisTickColor }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[0, 1]}
                  hide
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme === 'dark' ? '#0F172A' : '#FFFFFF',
                    borderColor: theme === 'dark' ? 'rgba(148,163,184,0.25)' : 'rgba(15,23,42,0.12)',
                    borderRadius: 10,
                    fontSize: 11,
                  }}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="bestValue"
                  name="Best pChEMBL"
                  stroke={theme === 'dark' ? '#34D399' : '#059669'}
                  strokeWidth={2.2}
                  dot={{ r: 3 }}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="conformalCoverage"
                  name="Conformal Cov."
                  stroke={theme === 'dark' ? '#F472B6' : '#DB2777'}
                  strokeWidth={1.8}
                  strokeDasharray="4 3"
                  dot={{ r: 2.5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-between font-mono-num text-[10px] text-[var(--text-muted)]">
            <span>
              Best:{' '}
              <strong className="text-emerald-600 dark:text-emerald-400">
                {runResult.dmtaTrace[runResult.dmtaTrace.length - 1].bestValue.toFixed(2)}
              </strong>
            </span>
            <span>
              Recall:{' '}
              <strong className="text-cyan-600 dark:text-cyan-400">
                {(runResult.dmtaTrace[runResult.dmtaTrace.length - 1].topHitRecall * 100).toFixed(0)}%
              </strong>
            </span>
            <span>
              90% Cov:{' '}
              <strong className="text-pink-600 dark:text-pink-400">
                {(runResult.dmtaTrace[runResult.dmtaTrace.length - 1].conformalCoverage * 100).toFixed(0)}%
              </strong>
            </span>
          </div>
        </div>
      )}

      {/* Output Verified Molecules Gallery */}
      {runResult.molecules.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold flex items-center gap-1">
              <Sparkles className="h-3 w-3 text-emerald-500" /> Verified In Silico Leads
            </span>
            <span className="font-mono-num text-[10px] text-[var(--text-muted)]">
              Top {runResult.molecules.length}
            </span>
          </div>

          <div className="space-y-2">
            {runResult.molecules.map((mol) => {
              const matched = molSvgMap.get(mol.smiles);
              const svg = matched
                ? theme === 'dark'
                  ? matched.svg_mini_dark
                  : matched.svg_mini_light
                : null;

              return (
                <div
                  key={mol.id}
                  className="rounded-xl border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] p-2.5 space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-[var(--text-primary)] truncate">
                        {mol.name}
                      </div>
                      {mol.metrics.reaction_or_transform && (
                        <div className="font-mono-num text-[9.5px] text-amber-600 dark:text-amber-300 truncate">
                          {mol.metrics.reaction_or_transform}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => onInspectIn3D(mol.target)}
                      className="inline-flex items-center gap-0.5 px-2 py-1 rounded-lg text-[10px] font-semibold bg-cyan-500/12 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/22 shrink-0 cursor-pointer"
                    >
                      3D <ArrowUpRight className="h-3 w-3" />
                    </button>
                  </div>

                  {svg && (
                    <div
                      className="rounded-lg bg-[var(--bg-elevated)]/70 border border-[var(--circuit-node-border)]/70 h-20 flex items-center justify-center p-1 overflow-hidden rdkit-svg-wrap"
                      dangerouslySetInnerHTML={{ __html: svg }}
                    />
                  )}

                  <div className="grid grid-cols-4 gap-1 text-center font-mono-num text-[9.5px]">
                    <div className="rounded-md bg-[var(--bg-elevated)] py-1">
                      <div className="text-[8.5px] text-[var(--text-muted)]">Vina ΔG</div>
                      <div className="font-bold text-cyan-600 dark:text-cyan-400">
                        {mol.metrics.vina_dg != null ? mol.metrics.vina_dg.toFixed(1) : '—'}
                      </div>
                    </div>
                    <div className="rounded-md bg-[var(--bg-elevated)] py-1">
                      <div className="text-[8.5px] text-[var(--text-muted)]">MPO v2</div>
                      <div className="font-bold text-emerald-600 dark:text-emerald-400">
                        {mol.metrics.cns_mpo_v2 != null ? mol.metrics.cns_mpo_v2.toFixed(2) : '—'}
                      </div>
                    </div>
                    <div className="rounded-md bg-[var(--bg-elevated)] py-1">
                      <div className="text-[8.5px] text-[var(--text-muted)]">hERG</div>
                      <div
                        className={`font-bold ${
                          (mol.metrics.herg_prob ?? 0) <= 0.3
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-rose-500'
                        }`}
                      >
                        {mol.metrics.herg_prob != null
                          ? `${(mol.metrics.herg_prob * 100).toFixed(0)}%`
                          : '—'}
                      </div>
                    </div>
                    <div className="rounded-md bg-[var(--bg-elevated)] py-1">
                      <div className="text-[8.5px] text-[var(--text-muted)]">Reality</div>
                      <div className="font-bold text-violet-600 dark:text-violet-400">
                        {mol.metrics.reality_score != null
                          ? mol.metrics.reality_score.toFixed(2)
                          : '—'}
                      </div>
                    </div>
                  </div>

                  {mol.metrics.conformal_lo != null && mol.metrics.conformal_hi != null && (
                    <div className="font-mono-num text-[9.5px] text-pink-600 dark:text-pink-300 bg-pink-500/8 rounded-md px-2 py-1">
                      90% Split-Conformal: [{mol.metrics.conformal_lo.toFixed(2)},{' '}
                      {mol.metrics.conformal_hi.toFixed(2)}]
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => handleCopy(mol.id, mol.smiles)}
                    className="w-full px-2 py-1 rounded-md bg-[var(--bg-elevated)]/80 hover:bg-[var(--bg-elevated)] font-mono-num text-[9.5px] text-[var(--text-secondary)] flex items-center justify-between gap-1.5 cursor-pointer"
                  >
                    <span className="truncate">{mol.smiles}</span>
                    {copiedId === mol.id ? (
                      <Check className="h-3 w-3 text-emerald-500 shrink-0" />
                    ) : (
                      <Copy className="h-3 w-3 text-[var(--text-muted)] shrink-0" />
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Clickable Step Execution Timeline */}
      <div className="space-y-1.5 pt-2 border-t border-[var(--circuit-node-border)]">
        <div className="text-[10.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
          Step Telemetry (Round {runResult.roundsCompleted})
        </div>
        <div className="space-y-1">
          {latestSteps.map((step) => {
            const def = BLOCK_MAP[step.blockType];
            return (
              <button
                key={`${step.round}-${step.nodeId}`}
                type="button"
                onClick={() => onFocusNode(step.nodeId)}
                className="w-full text-left px-2.5 py-1.5 rounded-lg border border-[var(--circuit-node-border)] bg-[var(--bg-surface)] hover:border-cyan-500/45 transition-colors flex items-center justify-between gap-2 cursor-pointer"
              >
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold text-[var(--text-primary)] truncate">
                    {def?.shortName ?? step.blockType}
                  </div>
                  <div className="font-mono-num text-[9.5px] text-emerald-600 dark:text-emerald-400 truncate">
                    {step.summary}
                  </div>
                </div>
                <span className="font-mono-num text-[10px] text-[var(--text-muted)] shrink-0">
                  {step.durationMs}ms
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
