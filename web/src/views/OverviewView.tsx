import {
  Activity,
  ArrowRight,
  Atom,
  BarChart3,
  CheckCircle2,
  Cpu,
  FlaskConical,
  GitBranch,
  Layers,
  ShieldCheck,
  Sparkles,
  Workflow,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { Mol3DViewer } from '../components/Mol3DViewer';
import { loadComplexDetail } from '../lib/data';
import type { ComplexDetail, OverviewData, StudioTab, ThemeMode } from '../types/studio';

interface OverviewViewProps {
  overview: OverviewData;
  theme: ThemeMode;
  onNavigate: (tab: StudioTab, target?: string) => void;
}

const HERO_TARGETS = [
  { id: 'DRD1', label: 'DRD1 (9LLG)', tag: 'Tourette Anti-EPS' },
  { id: 'ACHE', label: 'ACHE (4EY7)', tag: "Alzheimer's v4.0" },
  { id: 'MAOB', label: 'MAOB (2V5Z)', tag: "Alzheimer's +FAD" },
  { id: 'GRM5', label: 'GRM5 (4OO9)', tag: 'Tourette CSTC' },
  { id: 'CHRM4', label: 'CHRM4 (8FX5)', tag: 'Non-D2 M4' },
] as const;

const PIPELINE_STAGES = [
  {
    id: 'encoder',
    step: '01',
    title: 'Chiral PaiNN-SE3 3D Encoder',
    badge: 'SO(3) < 8e-6',
    accent: 'from-cyan-500 to-sky-500',
    summary:
      'Float32 SE(3)-equivariant message passing over 10 Å pocket heavy-atom + cofactor graphs (FAD/ADP/Zn/Mg).',
    details: [
      'Scalar invariants s ∈ ℝ¹⁹² + l=1 vector channels v ∈ ℝ⁶⁴ˣ³ over 5.5 Å radius-kNN graphs (24 Bessel RBFs).',
      'Preserves catalytic cofactors (FAD in MAO-B, ADP in NLRP3, Zn/Mg in PDE1B) while masking organic ligand leaks.',
      'Frozen bit-for-bit across v3.0–v4.0: retains CASF-2016 pKd r = 0.746 and DrugCLIP R@5 = 62.1% with 0.0 drift.',
    ],
  },
  {
    id: 'bridge',
    step: '02',
    title: 'PocketPerceiverResampler Bridge',
    badge: '7.12× CNS Attn',
    accent: 'from-indigo-500 to-violet-500',
    summary: 'Permutation- & SE(3)-invariant compression into 24 scale-matched soft prefix tokens.',
    details: [
      '2-layer 8-head cross-attention extracts 16 contact-focused queries + 4 PocketCVAE + 4 global/pKd tokens.',
      'Exact 0.0 attention on padded atoms and ScaleMatchRMSNorm eliminating Step-0 activation shock.',
      'Achieves 7.12× <4.5 Å crystal contact enrichment and 4.25× canonical CNS hotspot enrichment.',
    ],
  },
  {
    id: 'backbone',
    step: '03',
    title: '3 Local LLMs + 1.79 MB Browser AI',
    badge: 'SAFE-GPT · NovoMolGen · GP-MoLF',
    accent: 'from-emerald-500 to-teal-500',
    summary:
      'Multi-backbone chemical foundation transformers enabling de novo generation AND 3D scaffold rescue.',
    details: [
      'SAFE-GPT-87M (fragment SAFE + 3D clinical scaffold decoration), NovoMolGen-157M (MPS SDPA, IntDiv1=0.892), and GP-MoLFormer-Uniq.',
      '12.5× exact fast_4d_tril causal attention kernel + 11.15× KV-cached Classifier-Free Guidance (w=1.3).',
      'Distilled into a 1.79 MB INT8 KV-cached client-side Transformer running live in your browser (View 6).',
    ],
  },
  {
    id: 'alignment',
    step: '04',
    title: '4-Ch GDPO + L_XP-DPO + 12-Ep Avoid-ome',
    badge: '5.5× Safe · 0% DCU',
    accent: 'from-amber-500 to-rose-500',
    summary:
      'Multiplicative safety-gated alignment across 12 industrial DMPK/Avoid-ome oracles & Split-Conformal UQ.',
    details: [
      'SAFE-GPT-87M + 4-Ch GDPO lifts 12-endpoint Pharma-Safe 5.5× vs v3.1 (6.8% → 37.3%, 22/24 Holm) with 0.0% DCU and 0.0% PDE10A sink.',
      'Slashes Biogen MDR1-MDCK efflux 7.6× (0.448 → 0.059) and validates external Rat Kp,uu (ρ = +0.724, p = 3.85×10⁻⁴⁰).',
      'Anti-Goodhart discipline: AutoDock Vina v1.2.7 is 100% held out from training, verifying 100% PoseBusters & FDA/EMA 10/10 dossier.',
    ],
  },
] as const;

export function OverviewView({ overview, theme, onNavigate }: OverviewViewProps) {
  const kpi = overview.headline_kpis;
  const [heroTarget, setHeroTarget] = useState<string>('DRD1');
  const [heroComplex, setHeroComplex] = useState<ComplexDetail | null>(null);
  const [activeStage, setActiveStage] = useState<string>('encoder');

  useEffect(() => {
    let alive = true;
    loadComplexDetail(heroTarget)
      .then((data) => {
        if (alive) setHeroComplex(data);
      })
      .catch(() => {
        // handled by viewer fallback
      });
    return () => {
      alive = false;
    };
  }, [heroTarget]);

  const currentStage =
    PIPELINE_STAGES.find((s) => s.id === activeStage) ?? PIPELINE_STAGES[0];

  const heroPills = [
    { label: 'Human 3D Targets', value: `${kpi.n_targets}` },
    { label: 'Local LLMs + Web AI', value: `${kpi.n_local_llms ?? 3} + 1` },
    { label: 'v4 Pharma-Safe', value: `${kpi.v4_safe_fold_vs_v31 ?? 5.5}× (${kpi.v4_safegpt_safe_pct ?? 37.3}%)` },
    { label: 'Avoid-ome Oracles', value: `${kpi.v4_n_avoidome_endpoints ?? 12}-Ep` },
    { label: 'PoseBusters', value: `${kpi.posebusters_pass_pct.toFixed(0)}%` },
    { label: 'DCU / Sink', value: `${(kpi.v4_dcu_pct ?? 0).toFixed(0)}%` },
  ];

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6 sm:py-8 space-y-10">
      {/* Split Executive Hero */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left: Scientific Positioning + Live KPIs + CTAs */}
        <div className="lg:col-span-6 flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border border-cyan-500/25">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              GenEuclidean {overview.version} · {kpi.n_targets} Human 3D Targets (Tourette-GABA × Schizophrenia × Alzheimer&apos;s)
            </div>

            <h1 className="text-3xl sm:text-4xl xl:text-[44px] font-extrabold tracking-tight leading-[1.08] text-[var(--text-primary)]">
              3D <span className="bg-gradient-to-r from-cyan-400 via-sky-400 to-emerald-400 bg-clip-text text-transparent">SE(3)-Equivariant</span>{' '}
              Tri-Disorder Drug Discovery Studio
            </h1>

            <p className="text-sm sm:text-base text-[var(--text-secondary)] leading-relaxed max-w-2xl">
              Uniting frozen chiral <code className="font-mono-num text-cyan-600 dark:text-cyan-300">PaiNN-SE3</code>{' '}
              3D pocket geometry with three local chemical foundation LLMs (
              <strong className="text-[var(--text-primary)]">SAFE-GPT-87M</strong>,{' '}
              <strong className="text-[var(--text-primary)]">NovoMolGen-157M</strong>,{' '}
              <strong className="text-[var(--text-primary)]">GP-MoLFormer</strong>),{' '}
              <strong>4-Channel GDPO + L_XP-DPO</strong>, 12-endpoint industrial Avoid-ome/DMPK oracles,
              and a live <strong>1.79 MB in-browser neural generator</strong>.
            </p>

            {/* Live verified v4.0 KPI pills */}
            <div className="flex flex-wrap gap-2 pt-1">
              {heroPills.map((p) => (
                <div
                  key={p.label}
                  className="px-3 py-1.5 rounded-xl lab-card flex items-center gap-2 text-xs"
                >
                  <span className="text-[var(--text-muted)] font-medium">{p.label}</span>
                  <span className="font-mono-num font-bold text-[var(--text-primary)]">{p.value}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Primary CTAs */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="button"
              onClick={() => onNavigate('studio3d', heroTarget)}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-lg shadow-cyan-500/20 hover:brightness-110 transition-all cursor-pointer"
            >
              <Atom className="h-4 w-4" />
              Explore {kpi.n_targets} 3D Complexes
              <ArrowRight className="h-4 w-4" />
            </button>

            <button
              type="button"
              onClick={() => onNavigate('circuit')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm bg-cyan-500/15 text-cyan-700 dark:text-cyan-200 border border-cyan-500/35 hover:bg-cyan-500/25 transition-all cursor-pointer"
            >
              <Workflow className="h-4 w-4" />
              Circuit Studio
            </button>

            <button
              type="button"
              onClick={() => onNavigate('generator')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/35 hover:bg-emerald-500/25 transition-all cursor-pointer"
            >
              <Sparkles className="h-4 w-4" />
              Live Browser AI Generator
            </button>

            <button
              type="button"
              onClick={() => onNavigate('experiments')}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm lab-card text-[var(--text-primary)] hover:border-cyan-500/40 cursor-pointer"
            >
              <BarChart3 className="h-4 w-4 text-cyan-500" />
              v4.0 3-LLM Benchmarks
            </button>
          </div>

          {/* 4 Headline Metric Stat Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="lab-card rounded-xl p-3.5">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                v4 SAFE-GPT Safe
              </div>
              <div className="font-mono-num text-2xl font-bold text-emerald-500 dark:text-emerald-400 mt-1">
                {(kpi.v4_safegpt_safe_pct ?? 37.3).toFixed(1)}%
              </div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                {kpi.v4_safe_fold_vs_v31 ?? 5.5}× v3.1 · 0% DCU/sink
              </div>
            </div>

            <div className="lab-card rounded-xl p-3.5">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                Rat Kp,uu Calib.
              </div>
              <div className="font-mono-num text-2xl font-bold text-cyan-500 dark:text-cyan-400 mt-1">
                ρ={(kpi.v4_kpuu_rho ?? 0.724).toFixed(3)}
              </div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                n=239 · p=3.85×10⁻⁴⁰
              </div>
            </div>

            <div className="lab-card rounded-xl p-3.5">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                Donepezil Rescue
              </div>
              <div className="font-mono-num text-2xl font-bold text-amber-500 dark:text-amber-400 mt-1">
                −2.29
              </div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                ΔΔG vs BChE · hERG 0.95→0.14
              </div>
            </div>

            <div className="lab-card rounded-xl p-3.5">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                TS-GABA Gate-1
              </div>
              <div className="font-mono-num text-2xl font-bold text-violet-500 dark:text-violet-400 mt-1">
                {(kpi.ts_gaba_rmsd_median ?? 1.3).toFixed(2)} Å
              </div>
              <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                100% PB &amp; anchor · 4 pockets
              </div>
            </div>
          </div>
        </div>

        {/* Right: LIVE Interactive 3D WebGL Showcase */}
        <div className="lg:col-span-6 flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-[var(--text-secondary)]">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Live 3D WebGL Complex — Drag to Orbit · Scroll to Zoom
            </div>
            <div className="flex items-center gap-1 flex-wrap">
              {HERO_TARGETS.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => setHeroTarget(t.id)}
                  title={t.tag}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono-num font-semibold border transition-all cursor-pointer ${
                    heroTarget === t.id
                      ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                      : 'lab-card text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  {t.id}
                </button>
              ))}
            </div>
          </div>

          <Mol3DViewer
            complex={heroComplex}
            appTheme={theme}
            compact
            initialSpin
            className="h-[370px] sm:h-[430px] w-full shadow-2xl shadow-cyan-950/20"
          />

          {heroComplex && (
            <div className="lab-card rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                <span className="font-semibold text-[var(--text-primary)]">
                  {heroComplex.lead_id} · {heroComplex.disease_tag}
                </span>
                <span className="font-mono-num text-pink-500 dark:text-pink-400 font-semibold">
                  {heroComplex.physics_3d.anchor_label}
                  {heroComplex.physics_3d.anchor_min_dist !== null
                    ? ` ${heroComplex.physics_3d.anchor_min_dist.toFixed(2)} Å`
                    : ''}
                </span>
                <span className="font-mono-num text-[var(--text-muted)]">
                  MPO {heroComplex.admet.cns_mpo_v2.toFixed(2)} · P(hERG){' '}
                  {(heroComplex.admet.herg_prob * 100).toFixed(0)}% · PB PASS
                </span>
              </div>
              <button
                type="button"
                onClick={() => onNavigate('studio3d', heroComplex.target)}
                className="text-cyan-600 dark:text-cyan-400 font-semibold hover:underline flex items-center gap-1 cursor-pointer"
              >
                Inspect full 3D/2D <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Interactive 4-Stage Architecture Pipeline */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="text-[11px] font-mono-num uppercase tracking-widest text-cyan-600 dark:text-cyan-400 font-semibold">
              System Architecture
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-[var(--text-primary)]">
              4-Stage SE(3) Geometric 3D-to-LLM Drug Discovery Pipeline
            </h2>
          </div>
          <span className="text-xs text-[var(--text-muted)]">
            Click any stage to inspect mathematical invariants &amp; systems telemetry
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {PIPELINE_STAGES.map((stage) => {
            const active = stage.id === activeStage;
            return (
              <button
                key={stage.id}
                type="button"
                onClick={() => setActiveStage(stage.id)}
                className={`text-left rounded-2xl p-4 border transition-all cursor-pointer ${
                  active
                    ? 'bg-cyan-500/10 border-cyan-500/50 shadow-md shadow-cyan-500/10'
                    : 'lab-card hover:border-cyan-500/30'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-2.5">
                  <span
                    className={`font-mono-num text-[11px] font-bold px-2 py-0.5 rounded-md bg-gradient-to-r ${stage.accent} text-slate-950`}
                  >
                    STAGE {stage.step}
                  </span>
                  <span className="font-mono-num text-[10px] font-semibold text-[var(--text-muted)]">
                    {stage.badge}
                  </span>
                </div>
                <h3 className="font-bold text-sm text-[var(--text-primary)]">{stage.title}</h3>
                <p className="text-xs text-[var(--text-secondary)] mt-1.5 leading-relaxed">
                  {stage.summary}
                </p>
              </button>
            );
          })}
        </div>

        <div className="lab-card rounded-2xl p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="space-y-2 max-w-4xl">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-cyan-500" />
              <span className="font-bold text-sm text-[var(--text-primary)]">
                Stage {currentStage.step}: {currentStage.title}
              </span>
              <span className="font-mono-num text-[11px] px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300">
                {currentStage.badge}
              </span>
            </div>
            <ul className="space-y-1.5 text-xs sm:text-sm text-[var(--text-secondary)]">
              {currentStage.details.map((line) => (
                <li key={line} className="flex items-start gap-2">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0 mt-0.5" />
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          </div>
          <button
            type="button"
            onClick={() => onNavigate('experiments')}
            className="shrink-0 inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 cursor-pointer"
          >
            View v4.0 Benchmarks <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </section>

      {/* Version Evolution Timeline (v1.0 -> v4.0.0 LIVE) */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="text-[11px] font-mono-num uppercase tracking-widest text-emerald-600 dark:text-emerald-400 font-semibold">
              Research Lineage
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-[var(--text-primary)]">
              Evolution Timeline: 2021 ETH Thesis → v4.0.0 Tri-Disorder 3-LLM Studio
            </h2>
          </div>
          <span className="inline-flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
            <GitBranch className="h-3.5 w-3.5" /> All historical milestones tagged in Git
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
          {overview.evolution.map((stage) => {
            const isCurrent = stage.status === 'active' || stage.status === 'live';
            return (
              <div
                key={stage.version}
                className={`rounded-2xl p-4 border flex flex-col justify-between gap-3 ${
                  isCurrent
                    ? 'bg-gradient-to-b from-cyan-500/14 to-emerald-500/10 border-cyan-500/55 shadow-md shadow-cyan-500/10'
                    : 'lab-card'
                }`}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono-num text-sm font-extrabold text-[var(--text-primary)]">
                      {stage.version}
                    </span>
                    <span
                      className={`font-mono-num text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                        isCurrent
                          ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-300'
                          : 'bg-slate-500/15 text-[var(--text-muted)]'
                      }`}
                    >
                      {isCurrent ? 'LIVE' : stage.year}
                    </span>
                  </div>
                  <div className="font-bold text-xs sm:text-sm text-[var(--text-primary)]">
                    {stage.title}
                  </div>
                  <div className="font-mono-num text-[10.5px] text-cyan-600 dark:text-cyan-400">
                    {stage.architecture}
                  </div>
                  <ul className="space-y-1.5 pt-1 text-[11.5px] text-[var(--text-secondary)]">
                    {stage.highlights.map((h) => (
                      <li key={h} className="leading-snug flex items-start gap-1.5">
                        <span className="text-cyan-500 mt-0.5">•</span>
                        <span>{h}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="font-mono-num text-[10px] text-[var(--text-muted)] pt-2 border-t border-[var(--border-subtle)]">
                  tag: {stage.tag}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Four Scientific Breakthrough Highlights */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          {
            icon: ShieldCheck,
            color: 'text-emerald-500',
            title: 'SAFE-GPT 5.5× Pharma-Safe',
            body: '4-Ch GDPO + L_XP-DPO hits 37.3% 12-endpoint Pharma-Safe (22/24 Holm), 0.0% DCU, 0.0% PDE10A sink, and slashes MDR1 efflux 7.6×.',
          },
          {
            icon: Sparkles,
            color: 'text-cyan-500',
            title: '3D Clinical Scaffold Rescues',
            body: 'Decorates Donepezil (hERG 0.95→0.14, ΔΔG −2.29), Safinamide (ΔΔG −3.47), Xanomeline (M3 0.97→0.22), and Ecopipam (ΔΔG −0.98).',
          },
          {
            icon: Activity,
            color: 'text-violet-500',
            title: '12-Ep Avoid-ome + Kp,uu',
            body: '5-fold scaffold-CV oracles + 90% Split-Conformal UQ + external Rat Kp,uu ρ=+0.724 (p=3.85×10⁻⁴⁰) + FDA/EMA 10/10 dossier.',
          },
          {
            icon: Layers,
            color: 'text-amber-500',
            title: 'Tourette CSTC/GABA + Web AI',
            body: '4 new basal-ganglia pockets (GABRA5, GAT1, ADORA2A, GRM5; 1.30 Å redock) + 1.79 MB INT8 neural generator running live in-browser.',
          },
        ].map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.title} className="lab-card rounded-2xl p-4 space-y-2">
              <Icon className={`h-5 w-5 ${card.color}`} />
              <h3 className="font-bold text-sm text-[var(--text-primary)]">{card.title}</h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{card.body}</p>
            </div>
          );
        })}
      </section>

      {/* Explicit Translational In Silico Disclaimer */}
      <section className="rounded-2xl border border-amber-500/30 bg-amber-500/8 px-5 py-4 flex items-start gap-3">
        <FlaskConical className="h-5 w-5 text-amber-500 shrink-0 mt-0.5" />
        <div className="space-y-1 text-xs sm:text-sm">
          <div className="font-bold text-amber-700 dark:text-amber-300">
            Translational In Silico Research Notice
          </div>
          <p className="text-[var(--text-secondary)] leading-relaxed">{overview.disclaimer}</p>
        </div>
      </section>
    </div>
  );
}
