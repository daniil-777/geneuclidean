import {
  Atom,
  Check,
  Copy,
  Cpu,
  Download,
  FlaskConical,
  Play,
  ShieldCheck,
  Sliders,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  buildSteeringVector,
  loadBrowserGeneratorBundle,
  type BrowserChemicalTransformer,
  type GeneratorPreset,
  type PocketEmbeddingsPack,
} from '../lib/browserGenerator';
import { copyToClipboard, downloadTextFile } from '../lib/data';
import {
  loadBrowserChemScorer,
  type BrowserChemScorer,
  type LiveMoleculeScore,
} from '../lib/rdkitBrowser';
import type { MoleculeIndexEntry, ThemeMode } from '../types/studio';

interface GeneratorViewProps {
  molecules: MoleculeIndexEntry[];
  theme: ThemeMode;
  onInspectIn3D: (target: string) => void;
}

interface GeneratedCard {
  id: string;
  rawSmiles: string;
  latencyMs: number;
  tokenCount: number;
  score: LiveMoleculeScore | null;
}

export function GeneratorView({ molecules, theme, onInspectIn3D }: GeneratorViewProps) {
  const [transformer, setTransformer] = useState<BrowserChemicalTransformer | null>(null);
  const [pockets, setPockets] = useState<PocketEmbeddingsPack | null>(null);
  const [scorer, setScorer] = useState<BrowserChemScorer | null>(null);
  const [engineStatus, setEngineStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [statusMessage, setStatusMessage] = useState<string>('Checking quantized model pack…');

  const [mode, setMode] = useState<'single' | 'contrastive' | 'dual_blend'>('contrastive');
  const [primaryTarget, setPrimaryTarget] = useState<string>('DRD1');
  const [secondaryTarget, setSecondaryTarget] = useState<string>('DRD2');
  const [cfgScale, setCfgScale] = useState<number>(1.5);
  const [temperature, setTemperature] = useState<number>(0.85);
  const [topP, setTopP] = useState<number>(0.92);
  const [steerParam, setSteerParam] = useState<number>(0.5);
  const [batchSize, setBatchSize] = useState<number>(8);

  const [generating, setGenerating] = useState(false);
  const [streamPreview, setStreamPreview] = useState<string>('');
  const [results, setResults] = useState<GeneratedCard[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Auto-load neural generator + RDKit.js + calibrated ADMET/QSAR models on mount
  useEffect(() => {
    let alive = true;
    setEngineStatus('loading');
    setStatusMessage('Loading quantized INT8 3D-pocket Transformer & RDKit.js WASM…');

    Promise.all([loadBrowserGeneratorBundle(), loadBrowserChemScorer()])
      .then(([genBundle, chemScorer]) => {
        if (!alive) return;
        setTransformer(genBundle.transformer);
        setPockets(genBundle.pockets);
        setScorer(chemScorer);
        setEngineStatus('ready');
        setStatusMessage(
          `Ready · ${(genBundle.transformer.manifest.weights_bytes / (1024 * 1024)).toFixed(2)} MB INT8 Transformer + RDKit.js + Calibrated CNS Oracles`,
        );
      })
      .catch((err) => {
        if (!alive) return;
        setEngineStatus('unavailable');
        setStatusMessage(
          err instanceof Error
            ? err.message
            : 'Quantized browser model pack not yet present in web/public/models/.',
        );
      });

    return () => {
      alive = false;
    };
  }, []);

  const targetIds = useMemo(() => {
    if (pockets) return Object.keys(pockets.targets);
    return molecules.map((m) => m.target);
  }, [pockets, molecules]);

  const applyPreset = (preset: GeneratorPreset) => {
    setMode(preset.mode);
    setPrimaryTarget(preset.primary_target);
    if (preset.anti_target) setSecondaryTarget(preset.anti_target);
    else if (preset.secondary_target) setSecondaryTarget(preset.secondary_target);
    if (preset.alpha !== undefined) setSteerParam(preset.alpha);
    else if (preset.blend_t !== undefined) setSteerParam(preset.blend_t);
    setCfgScale(preset.cfg_scale);
    setTemperature(preset.temperature);
  };

  const handleGenerate = async () => {
    if (!transformer || !pockets || !scorer || generating) return;
    setGenerating(true);
    setResults([]);
    setStreamPreview('');

    const zCond = buildSteeringVector(
      pockets,
      mode,
      primaryTarget,
      secondaryTarget,
      steerParam,
    );
    const seenCanonical = new Set<string>();
    const generated: GeneratedCard[] = [];

    for (let i = 0; i < batchSize; i++) {
      // Yield to browser paint frame so live streaming tokens render smoothly
      await new Promise((r) => setTimeout(r, 8));
      const sample = transformer.generateOne({
        zCond,
        cfgScale,
        temperature,
        topK: transformer.manifest.defaults.top_k ?? 40,
        topP,
        maxNewTokens: transformer.manifest.defaults.max_new_tokens ?? 92,
        onToken: (partial) => setStreamPreview(partial),
      });
      const score = scorer.score(sample.smiles);
      if (score && seenCanonical.has(score.canonicalSmiles)) {
        // Keep unique molecules prioritized, still append if needed
        continue;
      }
      if (score) seenCanonical.add(score.canonicalSmiles);
      generated.push({
        id: `${Date.now()}-${i}`,
        rawSmiles: sample.smiles,
        latencyMs: sample.latencyMs,
        tokenCount: sample.tokenCount,
        score,
      });
      setResults([...generated]);
    }

    setStreamPreview('');
    setGenerating(false);
  };

  const summaryStats = useMemo(() => {
    if (results.length === 0) return null;
    const valid = results.filter((r) => r.score !== null);
    const avg = (arr: number[]) =>
      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    return {
      total: results.length,
      validCount: valid.length,
      validPct: (valid.length / results.length) * 100,
      safeCount: valid.filter((r) => r.score?.jointCnsSafe).length,
      meanMpo: avg(valid.map((r) => r.score!.cnsMpoV2)),
      meanBbb: avg(valid.map((r) => r.score!.bbbProb)) * 100,
      meanHerg: avg(valid.map((r) => r.score!.hergProb)) * 100,
      meanLatency: avg(results.map((r) => r.latencyMs)),
    };
  }, [results]);

  const handleCopy = async (id: string, smi: string) => {
    if (await copyToClipboard(smi)) {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1400);
    }
  };

  const handleDownloadBatch = () => {
    const lines = results
      .filter((r) => r.score)
      .map(
        (r, i) =>
          `${r.score!.canonicalSmiles}\tGenEuclidean_${primaryTarget}_live_${i + 1}\tMPO=${r.score!.cnsMpoV2.toFixed(2)}\tBBB=${r.score!.bbbProb.toFixed(2)}\thERG=${r.score!.hergProb.toFixed(2)}`,
      );
    if (lines.length === 0) return;
    downloadTextFile(
      `geneuclidean_live_${primaryTarget}_${mode}.smi`,
      lines.join('\n') + '\n',
      'text/plain',
    );
  };

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6 sm:py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="inline-flex items-center gap-1.5 text-[11px] font-mono-num uppercase tracking-widest text-emerald-600 dark:text-emerald-400 font-semibold">
            <Sparkles className="h-3.5 w-3.5" /> Zero-Backend Client-Side Neural Inference
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-primary)]">
            Live AI De Novo Molecule Generator
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
            Run GenEuclidean&apos;s real INT8 KV-cached 3D pocket-conditioned Transformer inside your
            browser using frozen chiral <code className="font-mono-num">PaiNN-SE3</code> pocket
            embeddings, live <strong>RDKit.js WASM</strong> 2D depictions, Pfizer CNS MPO v2, and
            5-fold scaffold-CV calibrated BBB/P-gp/hERG &amp; 12-target QSAR oracles.
          </p>
        </div>

        <div
          className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${
            engineStatus === 'ready'
              ? 'bg-emerald-500/12 border-emerald-500/35 text-emerald-700 dark:text-emerald-300'
              : engineStatus === 'loading'
                ? 'bg-cyan-500/12 border-cyan-500/35 text-cyan-700 dark:text-cyan-300'
                : 'bg-amber-500/12 border-amber-500/35 text-amber-700 dark:text-amber-300'
          }`}
        >
          <Cpu className={`h-3.5 w-3.5 ${engineStatus === 'loading' ? 'animate-spin' : ''}`} />
          <span>{statusMessage}</span>
        </div>
      </div>

      {/* Curated Clinical Task Presets */}
      {pockets && pockets.presets.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
            Curated Neuro-Discovery Steering Presets (1-Click)
          </div>
          <div className="flex flex-wrap gap-2">
            {pockets.presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset)}
                title={preset.description}
                className="px-3 py-1.5 rounded-xl lab-card hover:border-cyan-500/50 text-xs font-semibold text-[var(--text-primary)] flex items-center gap-1.5 cursor-pointer"
              >
                <Zap className="h-3.5 w-3.5 text-amber-500" />
                {preset.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT CONTROL PANEL */}
        <aside className="lg:col-span-4 lab-card rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 font-bold text-sm text-[var(--text-primary)]">
              <Sliders className="h-4 w-4 text-cyan-500" />
              3D Pocket Latent Steering
            </div>
            <span className="font-mono-num text-[10px] px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300">
              PaiNN-SE3 256-D
            </span>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[var(--text-secondary)]">
              Conditioning Mode
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {(
                [
                  ['single', 'Single Pocket'],
                  ['contrastive', 'A − α·B Anti'],
                  ['dual_blend', 'Dual Blend'],
                ] as const
              ).map(([id, lbl]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMode(id)}
                  className={`px-2 py-1.5 rounded-lg text-[11px] font-semibold border cursor-pointer ${
                    mode === id
                      ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                      : 'border-[var(--border-subtle)] text-[var(--text-muted)]'
                  }`}
                >
                  {lbl}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[var(--text-secondary)]">
              Primary Human 3D Pocket (Target A)
            </label>
            <select
              value={primaryTarget}
              onChange={(e) => setPrimaryTarget(e.target.value)}
              className="w-full px-3 py-2 rounded-xl text-xs bg-[var(--bg-elevated)] border border-[var(--border-subtle)] text-[var(--text-primary)] outline-none focus:border-cyan-500"
            >
              {targetIds.map((tid) => {
                const pInfo = pockets?.targets[tid];
                const mInfo = molecules.find((m) => m.target === tid);
                return (
                  <option key={tid} value={tid}>
                    {tid} ({pInfo?.primary_pdb_id ?? mInfo?.pdb_id ?? ''}) —{' '}
                    {pInfo?.name ?? mInfo?.disease_tag ?? tid}
                  </option>
                );
              })}
            </select>
          </div>

          {mode !== 'single' && (
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-[var(--text-secondary)]">
                {mode === 'contrastive'
                  ? 'Anti-Target Off-Target Repulsion (Target B)'
                  : 'Secondary Polypharmacology Pocket (Target B)'}
              </label>
              <select
                value={secondaryTarget}
                onChange={(e) => setSecondaryTarget(e.target.value)}
                className="w-full px-3 py-2 rounded-xl text-xs bg-[var(--bg-elevated)] border border-[var(--border-subtle)] text-[var(--text-primary)] outline-none focus:border-cyan-500"
              >
                {targetIds
                  .filter((id) => id !== primaryTarget)
                  .map((tid) => (
                    <option key={tid} value={tid}>
                      {tid} — {pockets?.targets[tid]?.name ?? tid}
                    </option>
                  ))}
              </select>
            </div>
          )}

          {/* Sampling sliders */}
          <div className="space-y-3 pt-1">
            {mode !== 'single' && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[var(--text-secondary)]">
                    {mode === 'contrastive'
                      ? `Repulsion Strength α (${primaryTarget} − α·${secondaryTarget})`
                      : `Pocket Interpolation t (A ↔ B)`}
                  </span>
                  <span className="font-mono-num font-bold text-emerald-600 dark:text-emerald-400">
                    {steerParam.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={steerParam}
                  onChange={(e) => setSteerParam(Number(e.target.value))}
                  className="w-full accent-emerald-500"
                />
              </div>
            )}

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-[var(--text-secondary)]">
                  Classifier-Free Guidance (CFG w)
                </span>
                <span className="font-mono-num font-bold text-cyan-600 dark:text-cyan-400">
                  {cfgScale.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.5"
                max="2.5"
                step="0.1"
                value={cfgScale}
                onChange={(e) => setCfgScale(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[var(--text-secondary)]">Temp (T)</span>
                  <span className="font-mono-num font-bold text-[var(--text-primary)]">
                    {temperature.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.45"
                  max="1.25"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-[var(--text-secondary)]">Top-p</span>
                  <span className="font-mono-num font-bold text-[var(--text-primary)]">
                    {topP.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.7"
                  max="0.98"
                  step="0.02"
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  className="w-full accent-cyan-500"
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-[var(--text-secondary)]">Batch Size</span>
                <span className="font-mono-num font-bold text-[var(--text-primary)]">
                  {batchSize} molecules
                </span>
              </div>
              <div className="grid grid-cols-4 gap-1.5">
                {[4, 8, 12, 16].map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setBatchSize(n)}
                    className={`py-1 rounded-lg font-mono-num text-xs font-semibold border cursor-pointer ${
                      batchSize === n
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                        : 'border-[var(--border-subtle)] text-[var(--text-muted)]'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            disabled={engineStatus !== 'ready' || generating}
            onClick={handleGenerate}
            className={`w-full py-2.5 rounded-xl font-bold text-sm flex items-center justify-center gap-2 transition-all ${
              engineStatus === 'ready' && !generating
                ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-lg shadow-cyan-500/20 hover:brightness-110 cursor-pointer'
                : 'bg-slate-500/20 text-[var(--text-muted)] border border-[var(--border-subtle)] cursor-not-allowed'
            }`}
          >
            <Play className="h-4 w-4" />
            {generating
              ? `Synthesizing Live in Browser (${results.length}/${batchSize})…`
              : `Generate ${batchSize} Novel Molecules Live`}
          </button>

          {results.length > 0 && (
            <button
              type="button"
              onClick={handleDownloadBatch}
              className="w-full py-2 rounded-xl text-xs font-semibold lab-card hover:border-cyan-500/50 text-[var(--text-primary)] flex items-center justify-center gap-1.5 cursor-pointer"
            >
              <Download className="h-3.5 w-3.5 text-emerald-500" />
              Export Valid Batch (.smi)
            </button>
          )}

          <button
            type="button"
            onClick={() => onInspectIn3D(primaryTarget)}
            className="w-full py-2 rounded-xl text-xs font-semibold bg-cyan-500/12 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/20 flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Atom className="h-3.5 w-3.5" /> Inspect {primaryTarget} Pocket in 3D Studio
          </button>
        </aside>

        {/* RIGHT LIVE OUTPUT WORKSPACE */}
        <section className="lg:col-span-8 space-y-4">
          {/* Live streaming token bar */}
          {generating && streamPreview && (
            <div className="lab-card rounded-2xl px-4 py-3 border-cyan-500/45 flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-ping shrink-0" />
              <div className="min-w-0">
                <div className="text-[10px] uppercase tracking-wider text-cyan-500 font-bold">
                  Autoregressive CFG Token Stream
                </div>
                <div className="font-mono-num text-xs text-[var(--text-primary)] truncate">
                  {streamPreview}
                </div>
              </div>
            </div>
          )}

          {/* Batch telemetry bar */}
          {summaryStats && (
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
              {[
                ['RDKit Valid', `${summaryStats.validPct.toFixed(0)}% (${summaryStats.validCount}/${summaryStats.total})`, 'text-emerald-500'],
                ['CNS-Safe', `${summaryStats.safeCount}/${summaryStats.validCount}`, 'text-cyan-500'],
                ['Mean MPO v2', `${summaryStats.meanMpo.toFixed(2)} / 6`, 'text-amber-500'],
                ['Mean P(BBB+) / hERG', `${summaryStats.meanBbb.toFixed(0)}% / ${summaryStats.meanHerg.toFixed(0)}%`, 'text-violet-500'],
                ['Inference Speed', `${summaryStats.meanLatency.toFixed(1)} ms/mol`, 'text-[var(--text-primary)]'],
              ].map(([k, v, cls]) => (
                <div key={k} className="lab-card rounded-xl p-3">
                  <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                    {k}
                  </div>
                  <div className={`font-mono-num text-sm font-bold mt-0.5 ${cls}`}>{v}</div>
                </div>
              ))}
            </div>
          )}

          {/* Generated Molecule Cards Grid */}
          {results.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {results.map((item, idx) => {
                const sc = item.score;
                const pPrimary = sc?.targetProbs[primaryTarget];
                const pSecondary =
                  mode !== 'single' ? sc?.targetProbs[secondaryTarget] : sc?.targetProbs['DRD2'];
                return (
                  <div key={item.id} className="lab-card rounded-2xl p-4 space-y-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono-num text-xs font-bold text-[var(--text-primary)]">
                          DeNovo #{idx + 1}
                        </span>
                        {sc?.jointCnsSafe ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30">
                            <ShieldCheck className="h-3 w-3" /> CNS-Safe
                          </span>
                        ) : sc ? (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/15 text-amber-600 dark:text-amber-300">
                            Valid
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/15 text-rose-500">
                            Invalid SMILES
                          </span>
                        )}
                      </div>
                      <span className="font-mono-num text-[10px] text-[var(--text-muted)]">
                        {item.latencyMs.toFixed(0)} ms · {item.tokenCount} tok
                      </span>
                    </div>

                    {sc && (
                      <>
                        <div
                          className="rounded-xl bg-[var(--bg-elevated)]/75 border border-[var(--border-subtle)] p-2 h-36 flex items-center justify-center overflow-hidden rdkit-svg-wrap"
                          dangerouslySetInnerHTML={{
                            __html: theme === 'dark' ? sc.svgDark : sc.svgLight,
                          }}
                        />

                        <button
                          type="button"
                          onClick={() => handleCopy(item.id, sc.canonicalSmiles)}
                          className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)] hover:border-cyan-500/50 font-mono-num text-[10.5px] text-[var(--text-secondary)] flex items-center justify-between gap-2 cursor-pointer"
                        >
                          <span className="truncate">{sc.canonicalSmiles}</span>
                          {copiedId === item.id ? (
                            <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                          ) : (
                            <Copy className="h-3.5 w-3.5 text-[var(--text-muted)] shrink-0" />
                          )}
                        </button>

                        <div className="grid grid-cols-4 gap-1.5 text-center font-mono-num text-[10px]">
                          <div className="rounded-lg bg-[var(--bg-elevated)]/80 py-1 px-1">
                            <div className="text-[9px] text-[var(--text-muted)]">MPO v2</div>
                            <div className="font-bold text-cyan-500">{sc.cnsMpoV2.toFixed(2)}</div>
                          </div>
                          <div className="rounded-lg bg-[var(--bg-elevated)]/80 py-1 px-1">
                            <div className="text-[9px] text-[var(--text-muted)]">P(BBB+)</div>
                            <div className="font-bold text-emerald-500">
                              {(sc.bbbProb * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div className="rounded-lg bg-[var(--bg-elevated)]/80 py-1 px-1">
                            <div className="text-[9px] text-[var(--text-muted)]">P(P-gp)</div>
                            <div className="font-bold text-amber-500">
                              {(sc.pgpProb * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div className="rounded-lg bg-[var(--bg-elevated)]/80 py-1 px-1">
                            <div className="text-[9px] text-[var(--text-muted)]">P(hERG)</div>
                            <div
                              className={`font-bold ${
                                sc.hergProb <= 0.3 ? 'text-emerald-500' : 'text-rose-500'
                              }`}
                            >
                              {(sc.hergProb * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[var(--border-subtle)] font-mono-num text-[10px] text-[var(--text-muted)]">
                          <span>
                            MW {sc.mw.toFixed(0)} · LogP {sc.clogp.toFixed(1)} · TPSA{' '}
                            {sc.tpsa.toFixed(0)} · pKa {sc.mostBasicPka.toFixed(1)}
                          </span>
                          {pPrimary !== undefined && (
                            <span className="text-cyan-600 dark:text-cyan-400 font-semibold">
                              P({primaryTarget}) {(pPrimary * 100).toFixed(0)}%
                              {pSecondary !== undefined && primaryTarget !== 'DRD2'
                                ? ` · P(${mode !== 'single' ? secondaryTarget : 'DRD2'}) ${(pSecondary * 100).toFixed(0)}%`
                                : ''}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="lab-card rounded-2xl p-8 flex flex-col items-center justify-center text-center gap-4 min-h-[420px]">
              <div className="h-14 w-14 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/30 flex items-center justify-center">
                <FlaskConical className="h-7 w-7 text-cyan-500" />
              </div>
              <div className="space-y-1.5 max-w-lg">
                <h2 className="text-lg font-bold text-[var(--text-primary)]">
                  {engineStatus === 'ready'
                    ? 'Neural Engine Loaded — Click Generate to Synthesize Novel Molecules Live'
                    : engineStatus === 'loading'
                      ? 'Initializing In-Browser Neural Inference Engine…'
                      : 'Ready for Quantized Browser Model Pack'}
                </h2>
                <p className="text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed">
                  {engineStatus === 'ready'
                    ? `Select any human CNS target or preset (e.g. Tourette Anti-EPS DRD1 − α·DRD2), tune CFG scale & temperature, and run live 2-stream KV-cached inference directly on your CPU.`
                    : `The Pure-TypeScript Per-Row INT8 KV-Cached Causal Transformer + RDKit.js + Calibrated CNS ADMET/QSAR runtime is fully wired and waiting for web/public/models/ weights.`}
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
