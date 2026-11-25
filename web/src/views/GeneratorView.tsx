import {
  Atom,
  Check,
  Copy,
  Cpu,
  Download,
  FlaskConical,
  Play,
  Search,
  ShieldCheck,
  Sliders,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  buildSteeringVector,
  cosineSimilarity,
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
  label: string;
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

  const [activePresetId, setActivePresetId] = useState<string>('tourette_anti_eps');
  const [mode, setMode] = useState<'single' | 'contrastive' | 'dual_blend'>('contrastive');
  const [primaryTarget, setPrimaryTarget] = useState<string>('DRD1');
  const [secondaryTarget, setSecondaryTarget] = useState<string>('DRD2');
  const [cfgScale, setCfgScale] = useState<number>(1.35);
  const [temperature, setTemperature] = useState<number>(0.88);
  const [topP, setTopP] = useState<number>(0.92);
  const [steerParam, setSteerParam] = useState<number>(0.55);
  const [batchSize, setBatchSize] = useState<number>(8);

  const [generating, setGenerating] = useState(false);
  const [streamPreview, setStreamPreview] = useState<string>('');
  const [results, setResults] = useState<GeneratedCard[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [customSmiles, setCustomSmiles] = useState<string>('');
  const [customError, setCustomError] = useState<string | null>(null);

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
        const mb = (genBundle.transformer.manifest.weights_bytes / (1024 * 1024)).toFixed(2);
        setStatusMessage(
          `LIVE · ${mb} MB INT8 3D-Transformer + RDKit.js + 15 Calibrated Oracles`,
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

  // Live PaiNN-SE3 steering vector & cosine alignment readout
  const steeringTelemetry = useMemo(() => {
    if (!pockets) return null;
    const z = buildSteeringVector(pockets, mode, primaryTarget, secondaryTarget, steerParam);
    const zA = pockets.targets[primaryTarget]?.embedding;
    const zB = pockets.targets[secondaryTarget]?.embedding;
    return {
      z,
      cosA: zA ? cosineSimilarity(z, zA) : 0,
      cosB: zB ? cosineSimilarity(z, zB) : 0,
    };
  }, [pockets, mode, primaryTarget, secondaryTarget, steerParam]);

  const verificationOverall = useMemo(() => {
    const v = transformer?.manifest.verification as
      | { overall?: Record<string, number> }
      | undefined;
    return v?.overall ?? null;
  }, [transformer]);

  const applyPreset = (preset: GeneratorPreset) => {
    setActivePresetId(preset.id);
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
    if (!transformer || !pockets || !scorer || !steeringTelemetry || generating) return;
    setGenerating(true);
    setResults([]);
    setStreamPreview('');
    setCustomError(null);

    const zCond = steeringTelemetry.z;
    const seenCanonical = new Set<string>();
    const generated: GeneratedCard[] = [];

    for (let i = 0; i < batchSize; i++) {
      await new Promise((r) => setTimeout(r, 8));
      const sample = await transformer.generateOneAsync({
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
        continue;
      }
      if (score) seenCanonical.add(score.canonicalSmiles);
      generated.push({
        id: `${Date.now()}-${i}`,
        label: `DeNovo #${generated.length + 1}`,
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

  const handleScoreCustom = () => {
    if (!scorer) return;
    const smi = customSmiles.trim();
    if (!smi) return;
    const t0 = performance.now();
    const score = scorer.score(smi);
    if (!score) {
      setCustomError('Invalid SMILES string — RDKit.js could not parse molecule.');
      return;
    }
    setCustomError(null);
    setResults((prev) => [
      {
        id: `custom-${Date.now()}`,
        label: 'Custom SMILES',
        rawSmiles: smi,
        latencyMs: performance.now() - t0,
        tokenCount: smi.length,
        score,
      },
      ...prev,
    ]);
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
            <Sparkles className="h-3.5 w-3.5" /> Zero-Backend Client-Side Neural Inference (Pure TS INT8)
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-primary)]">
            Live AI De Novo Molecule Generator &amp; CNS Scorer
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
            Synthesize novel small molecules live in your browser using GenEuclidean&apos;s distilled
            1.69M-parameter INT8 KV-cached Transformer conditioned on frozen chiral{' '}
            <code className="font-mono-num">PaiNN-SE3</code> 3D pocket embeddings, scored in real time
            by <strong>RDKit.js WASM</strong> + 15 calibrated scaffold-CV CNS ADMET &amp; target QSAR oracles.
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

      {/* Verified Embedded Audit Strip from generator_manifest.json */}
      {verificationOverall && (
        <div className="lab-card rounded-2xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono-num">
          <span className="font-sans font-bold text-cyan-600 dark:text-cyan-400 flex items-center gap-1.5">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" />
            Verified Browser INT8 Benchmarks (12 Targets × N=64):
          </span>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[var(--text-secondary)]">
            <span>
              Valid <strong className="text-emerald-500">{(verificationOverall.raw_validity * 100).toFixed(1)}%</strong>
            </span>
            <span>
              Novel <strong className="text-cyan-500">{(verificationOverall.strict_novelty_vs_chembl_zinc * 100).toFixed(1)}%</strong>
            </span>
            <span>
              IntDiv1 <strong>{verificationOverall.intdiv1.toFixed(3)}</strong>
            </span>
            <span>
              MPO v2 <strong className="text-amber-500">{verificationOverall.mean_cns_mpo_v2.toFixed(2)}</strong>
            </span>
            <span>
              BBB <strong>{(verificationOverall.mean_bbb_full_oracle * 100).toFixed(0)}%</strong> · hERG{' '}
              <strong className="text-emerald-500">{(verificationOverall.mean_herg_full_oracle * 100).toFixed(0)}%</strong>
            </span>
          </div>
        </div>
      )}

      {/* Curated Clinical Task Presets */}
      {pockets && pockets.presets.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] font-semibold">
            Curated Neuro-Discovery Steering Presets (1-Click)
          </div>
          <div className="flex flex-wrap gap-2">
            {pockets.presets.map((preset) => {
              const active = preset.id === activePresetId;
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => applyPreset(preset)}
                  title={preset.description}
                  className={`px-3 py-1.5 rounded-xl border text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                    active
                      ? 'bg-cyan-500/18 border-cyan-500/55 text-cyan-700 dark:text-cyan-200 shadow-sm'
                      : 'lab-card hover:border-cyan-500/45 text-[var(--text-primary)]'
                  }`}
                >
                  <Zap className={`h-3.5 w-3.5 ${active ? 'text-cyan-400' : 'text-amber-500'}`} />
                  {preset.label}
                </button>
              );
            })}
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
                  onClick={() => {
                    setActivePresetId('');
                    setMode(id);
                  }}
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
              onChange={(e) => {
                setActivePresetId('');
                setPrimaryTarget(e.target.value);
              }}
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
                onChange={(e) => {
                  setActivePresetId('');
                  setSecondaryTarget(e.target.value);
                }}
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

          {/* Live PaiNN-SE3 hypersphere cosine readout */}
          {steeringTelemetry && (
            <div className="rounded-xl bg-[var(--bg-elevated)]/85 border border-[var(--border-subtle)] p-2.5 space-y-1.5 font-mono-num text-[10.5px]">
              <div className="flex justify-between">
                <span className="text-[var(--text-muted)]">cos(z_steer, {primaryTarget})</span>
                <span className="font-bold text-emerald-500">
                  {steeringTelemetry.cosA >= 0 ? '+' : ''}
                  {steeringTelemetry.cosA.toFixed(3)}
                </span>
              </div>
              {mode !== 'single' && (
                <div className="flex justify-between">
                  <span className="text-[var(--text-muted)]">cos(z_steer, {secondaryTarget})</span>
                  <span
                    className={`font-bold ${
                      mode === 'contrastive'
                        ? steeringTelemetry.cosB <= 0
                          ? 'text-cyan-500'
                          : 'text-amber-500'
                        : 'text-violet-500'
                    }`}
                  >
                    {steeringTelemetry.cosB >= 0 ? '+' : ''}
                    {steeringTelemetry.cosB.toFixed(3)}
                  </span>
                </div>
              )}
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
                step="0.05"
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
                  min="0.5"
                  max="1.25"
                  step="0.02"
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
              ? `Synthesizing Live (${results.length}/${batchSize})…`
              : `Generate ${batchSize} Novel Molecules Live`}
          </button>

          {/* Instant Custom SMILES Oracle Scorer */}
          <div className="pt-3 border-t border-[var(--border-subtle)] space-y-1.5">
            <label className="text-[11px] font-semibold text-[var(--text-secondary)] flex items-center gap-1.5">
              <Search className="h-3 w-3 text-cyan-500" />
              Score Any Custom SMILES (Live ADMET + 12-Target QSAR)
            </label>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={customSmiles}
                onChange={(e) => setCustomSmiles(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleScoreCustom()}
                placeholder="Paste SMILES (e.g. ecopipam)…"
                className="flex-1 min-w-0 px-2.5 py-1.5 rounded-lg text-xs font-mono-num bg-[var(--bg-elevated)] border border-[var(--border-subtle)] text-[var(--text-primary)] outline-none focus:border-cyan-500"
              />
              <button
                type="button"
                disabled={engineStatus !== 'ready'}
                onClick={handleScoreCustom}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold lab-card hover:border-cyan-500/50 text-[var(--text-primary)] disabled:opacity-40 cursor-pointer"
              >
                Score
              </button>
            </div>
            {customError && <p className="text-[10.5px] text-rose-500">{customError}</p>}
          </div>

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
                  Autoregressive 2-Stream CFG Token Generation
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
                [
                  'RDKit Valid',
                  `${summaryStats.validPct.toFixed(0)}% (${summaryStats.validCount}/${summaryStats.total})`,
                  'text-emerald-500',
                ],
                ['CNS-Safe', `${summaryStats.safeCount}/${summaryStats.validCount}`, 'text-cyan-500'],
                ['Mean MPO v2', `${summaryStats.meanMpo.toFixed(2)} / 6`, 'text-amber-500'],
                [
                  'BBB+ / hERG',
                  `${summaryStats.meanBbb.toFixed(0)}% / ${summaryStats.meanHerg.toFixed(0)}%`,
                  'text-violet-500',
                ],
                [
                  'CPU Speed',
                  `${summaryStats.meanLatency.toFixed(0)} ms/mol`,
                  'text-[var(--text-primary)]',
                ],
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
              {results.map((item) => {
                const sc = item.score;
                const pPrimary = sc?.targetProbs[primaryTarget];
                const compTgt = mode !== 'single' ? secondaryTarget : primaryTarget === 'DRD1' ? 'DRD2' : null;
                const pComp = compTgt ? sc?.targetProbs[compTgt] : undefined;
                const topTargets = sc
                  ? Object.entries(sc.targetProbs)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                  : [];

                return (
                  <div key={item.id} className="lab-card rounded-2xl p-4 space-y-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-mono-num text-xs font-bold text-[var(--text-primary)]">
                          {item.label}
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

                        {/* Live QSAR selectivity & top predicted targets */}
                        <div className="flex flex-wrap items-center justify-between gap-1.5 pt-1 border-t border-[var(--border-subtle)] font-mono-num text-[10px]">
                          <span className="text-[var(--text-muted)]">
                            MW {sc.mw.toFixed(0)} · LogP {sc.clogp.toFixed(1)} · TPSA{' '}
                            {sc.tpsa.toFixed(0)} · pKa {sc.mostBasicPka.toFixed(1)}
                          </span>
                          {pPrimary !== undefined && (
                            <span className="text-cyan-600 dark:text-cyan-400 font-bold">
                              {primaryTarget} {(pPrimary * 100).toFixed(0)}%
                              {compTgt && pComp !== undefined
                                ? ` · ${compTgt} ${(pComp * 100).toFixed(0)}%`
                                : ''}
                            </span>
                          )}
                        </div>

                        {topTargets.length > 0 && (
                          <div className="flex flex-wrap items-center gap-1 text-[9.5px] font-mono-num">
                            <span className="text-[var(--text-muted)]">Top QSAR:</span>
                            {topTargets.map(([t, p]) => (
                              <span
                                key={t}
                                className={`px-1.5 py-0.5 rounded ${
                                  t === primaryTarget
                                    ? 'bg-emerald-500/18 text-emerald-600 dark:text-emerald-300 font-bold'
                                    : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)]'
                                }`}
                              >
                                {t} {(p * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        )}
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
                    ? 'Neural Engine Ready — Click Generate to Synthesize Novel Molecules Live'
                    : engineStatus === 'loading'
                      ? 'Initializing In-Browser Neural Inference Engine…'
                      : 'Quantized Browser Model Pack Not Found'}
                </h2>
                <p className="text-xs sm:text-sm text-[var(--text-secondary)] leading-relaxed">
                  {engineStatus === 'ready'
                    ? `Pick any clinical steering preset above (like Tourette Anti-EPS DRD1 − α·DRD2), tune CFG scale & temperature, and watch novel CNS-safe ligands stream live with RDKit.js 2D depictions & 15 calibrated oracles.`
                    : `The Pure-TypeScript Per-Row INT8 KV-Cached Causal Transformer + RDKit.js runtime is waiting for web/public/models/ weights.`}
                </p>
              </div>
              {engineStatus === 'ready' && (
                <button
                  type="button"
                  onClick={handleGenerate}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-lg shadow-cyan-500/20 hover:brightness-110 cursor-pointer"
                >
                  <Play className="h-4 w-4" /> Synthesize {batchSize} {primaryTarget} Molecules Now
                </button>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
