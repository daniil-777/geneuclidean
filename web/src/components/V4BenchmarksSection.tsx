import {
  Activity,
  ArrowRight,
  Atom,
  Brain,
  CheckCircle2,
  Cpu,
  FlaskConical,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatPValue } from '../lib/data';
import type {
  ThemeMode,
  V4AvoidomeEndpoint,
  V4BenchmarksData,
} from '../types/studio';

interface V4BenchmarksSectionProps {
  v4: V4BenchmarksData;
  theme: ThemeMode;
  onInspectTarget: (target: string) => void;
}

const CATEGORY_COLORS: Record<V4AvoidomeEndpoint['category'], string> = {
  'BBB / Efflux': '#38BDF8',
  'DMPK / CYP': '#A78BFA',
  'Avoid-ome': '#10B981',
  Toxicology: '#F59E0B',
};

export function V4BenchmarksSection({
  v4,
  theme,
  onInspectTarget,
}: V4BenchmarksSectionProps) {
  const isDark = theme === 'dark';
  const gridStroke = isDark ? 'rgba(148,163,184,0.16)' : 'rgba(15,23,42,0.10)';
  const axisTick = isDark ? '#94A3B8' : '#475569';
  const tooltipStyle = {
    backgroundColor: isDark ? '#0B1224' : '#FFFFFF',
    borderColor: isDark ? 'rgba(148,163,184,0.28)' : 'rgba(15,23,42,0.16)',
    borderRadius: '12px',
    fontSize: '12px',
    color: isDark ? '#F8FAFC' : '#0F172A',
  };

  const arms = v4.llm_head_to_head.arms;
  const stats = v4.llm_head_to_head.safegpt_vs_v31_stats;
  const kpuu = v4.avoidome_12ep.kpuu_benchmark;
  const dossier = v4.credibility_dossier;

  const tsGabaChartData = v4.ts_circuit_gaba.pockets.map((p) => ({
    target: p.target,
    pdb: p.pdb_id,
    geneuclidean_strength: Math.abs(p.best_vina_dg),
    clinical_strength: Math.abs(p.clinical_ref_best_dg),
    zinc_strength: Math.abs(p.zinc_best_vina_dg),
    best_vina_dg: p.best_vina_dg,
    clinical_ref_best_dg: p.clinical_ref_best_dg,
    zinc_best_vina_dg: p.zinc_best_vina_dg,
  }));

  const bestSafe = Math.max(...arms.map((a) => a.v4_pharma_safe_pct));
  const bestDiv = Math.max(...arms.map((a) => a.int_div_1));
  const bestMpo = Math.max(...arms.map((a) => a.cns_mpo_v2));
  const minMdr1 = Math.min(...arms.map((a) => a.mdr1_efflux_pct));
  const minDcu = Math.min(...arms.map((a) => a.reagent_dcu_pct));

  return (
    <div className="space-y-6">
      {/* v4.0 Headline Verification Strip */}
      <div className="lab-card rounded-2xl p-4 sm:p-5 border-emerald-500/30 bg-gradient-to-r from-emerald-500/[0.06] via-cyan-500/[0.04] to-violet-500/[0.05]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-num font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30">
                <Sparkles className="h-3 w-3" /> GenEuclidean v4.0.0 Tri-Disorder Suite
              </span>
              <span className="text-[11px] font-mono-num text-[var(--text-muted)]">
                24 Tri-Disorder + 4 Tourette CSTC/GABA = 28 Human Targets · N = 1,536/arm
              </span>
            </div>
            <h2 className="text-lg sm:text-xl font-extrabold tracking-tight text-[var(--text-primary)]">
              3-LLM Head-to-Head, SAFE-GPT 3D Clinical Rescues, 12-Endpoint Avoid-ome &amp; CSTC-GABA
            </h2>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono-num">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30 font-semibold">
              SAFE-GPT +{stats.v4_safe_diff_pp.toFixed(1)} pp ({stats.v4_safe_holm_n}/24 Holm)
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 font-semibold">
              Gate 1 Exp: {v4.gate1_expansion.median_top1_rmsd.toFixed(2)} Å ·{' '}
              {v4.gate1_expansion.mean_pb_pct.toFixed(0)}% PB
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-500/30 font-semibold">
              {dossier.fda_ema_principles_total} guiding principles
            </span>
          </div>
        </div>
      </div>

      {/* PANEL A: 3-LLM Head-to-Head across 24 Tri-Disorder Targets */}
      <section className="lab-card rounded-2xl p-5 space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono-num uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold">
              <Cpu className="h-3.5 w-3.5" /> v4.0 Panel A · 3 Local Generative LLMs Head-to-Head (24 Targets)
            </div>
            <h3 className="font-bold text-base sm:text-lg text-[var(--text-primary)] mt-0.5">
              SAFE-GPT-87M vs. NovoMolGen-157M vs. GP-MoLFormer v4.0 vs. v3.1 &amp; ZINC
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              SAFE-GPT-87M lifts 12-endpoint v4 Pharma-Safe to{' '}
              <strong className="font-mono-num text-emerald-500">{bestSafe.toFixed(1)}%</strong>{' '}
              (5.5× v3.1, Wilcoxon p={formatPValue(stats.v4_safe_wilcoxon_p)}, {stats.v4_safe_holm_n}/24
              Holm), slashes MDR1 efflux 7.6× ({minMdr1.toFixed(1)}%), and eliminates both DCU
              reagents (0.0%) and PDE10A VMAT2-sink bleed (0.0%).
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-center">
          {/* Grouped Bar Chart */}
          <div className="xl:col-span-5 h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={arms} margin={{ top: 10, right: 8, left: -12, bottom: 6 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
                <XAxis dataKey="short_label" tick={{ fill: axisTick, fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fill: axisTick, fontSize: 10 }} unit="%" />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '10.5px' }} />
                <Bar
                  dataKey="v4_pharma_safe_pct"
                  name="12-Ep v4 Pharma-Safe (%)"
                  fill="#10B981"
                  radius={[5, 5, 0, 0]}
                />
                <Bar
                  dataKey="mdr1_efflux_pct"
                  name="MDR1 Efflux Risk (% ↓)"
                  fill="#F43F5E"
                  radius={[5, 5, 0, 0]}
                />
                <Bar
                  dataKey="triage_pass_pct"
                  name="MedChem Triage Pass (%)"
                  fill="#38BDF8"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Complete Head-to-Head Table */}
          <div className="xl:col-span-7 overflow-x-auto rounded-xl border border-[var(--border-subtle)]">
            <table className="w-full text-left text-xs whitespace-nowrap">
              <thead className="bg-[var(--bg-elevated)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                <tr>
                  <th className="py-2.5 px-3">Generative Arm</th>
                  <th className="py-2.5 px-2 text-right">Valid</th>
                  <th className="py-2.5 px-2 text-right">Novel</th>
                  <th className="py-2.5 px-2 text-right">IntDiv1</th>
                  <th className="py-2.5 px-2 text-right">v4 Safe ↑</th>
                  <th className="py-2.5 px-2 text-right">MDR1 ↓</th>
                  <th className="py-2.5 px-2 text-right">MPO v2 ↑</th>
                  <th className="py-2.5 px-2 text-right">DCU ↓</th>
                  <th className="py-2.5 px-2 text-right">PDE10A ↓</th>
                  <th className="py-2.5 px-2 text-right">Vina ΔG</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)] font-mono-num">
                {arms.map((arm) => {
                  const isBest = arm.key === v4.best_v4_arm;
                  return (
                    <tr
                      key={arm.key}
                      className={
                        isBest
                          ? 'bg-emerald-500/10 font-semibold'
                          : 'hover:bg-slate-500/6 transition-colors'
                      }
                    >
                      <td className="py-2 px-3 font-sans text-[var(--text-primary)] flex items-center gap-1.5">
                        <span>{arm.label}</span>
                        {isBest && (
                          <span className="text-[9px] font-mono-num px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-600 dark:text-emerald-300">
                            v4 Winner
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right text-[var(--text-secondary)]">
                        {arm.validity_pct.toFixed(1)}%
                      </td>
                      <td className="py-2 px-2 text-right text-[var(--text-secondary)]">
                        {arm.strict_novelty_pct.toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          Math.abs(arm.int_div_1 - bestDiv) < 1e-3
                            ? 'text-cyan-600 dark:text-cyan-400 font-bold'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {arm.int_div_1.toFixed(3)}
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          Math.abs(arm.v4_pharma_safe_pct - bestSafe) < 0.05
                            ? 'text-emerald-600 dark:text-emerald-400 font-extrabold'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {arm.v4_pharma_safe_pct.toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          Math.abs(arm.mdr1_efflux_pct - minMdr1) < 0.05
                            ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                            : arm.mdr1_efflux_pct > 30
                              ? 'text-rose-500'
                              : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {arm.mdr1_efflux_pct.toFixed(1)}%
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          Math.abs(arm.cns_mpo_v2 - bestMpo) < 0.01
                            ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {arm.cns_mpo_v2.toFixed(2)}
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          Math.abs(arm.reagent_dcu_pct - minDcu) < 0.01
                            ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                            : 'text-amber-500'
                        }`}
                      >
                        {arm.reagent_dcu_pct.toFixed(2)}%
                      </td>
                      <td
                        className={`py-2 px-2 text-right ${
                          arm.pde10a_vmat2_bleed_pct === 0
                            ? 'text-emerald-600 dark:text-emerald-400 font-bold'
                            : (arm.pde10a_vmat2_bleed_pct ?? 0) > 20
                              ? 'text-rose-500'
                              : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {arm.pde10a_vmat2_bleed_pct !== null
                          ? `${arm.pde10a_vmat2_bleed_pct.toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-2 px-2 text-right text-[var(--text-secondary)]">
                        {arm.mean_vina_dg !== null ? arm.mean_vina_dg.toFixed(2) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* PANEL B: SAFE-GPT 3D Clinical Scaffold Rescues */}
      <section className="lab-card rounded-2xl p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono-num uppercase tracking-wider text-cyan-600 dark:text-cyan-400 font-bold">
              <FlaskConical className="h-3.5 w-3.5" /> v4.0 Panel B · SAFE-GPT Fragment-Preserving Clinical Scaffold Rescue
            </div>
            <h3 className="font-bold text-base sm:text-lg text-[var(--text-primary)] mt-0.5">
              3D Counter-Target Verified Clinical Rescues (Alzheimer&apos;s, Schizophrenia &amp; Tourette)
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              SAFE fragment decoration preserves the clinical pharmacophore core while repairing
              liability hotspots (hERG cardiotoxicity, peripheral M3 toxicity, or subtype bleed) —
              validated via held-out on-target vs. off-target AutoDock Vina ΔΔG.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {v4.clinical_rescues.map((r) => {
            const parentSvg = isDark ? r.parent_svg_dark : r.parent_svg_light;
            const analogSvg = isDark ? r.analog_svg_dark : r.analog_svg_light;
            const hergDrop = r.parent_herg - r.analog_herg;
            const m3Drop = r.parent_chrm3 - r.analog_chrm3;
            return (
              <div
                key={r.id}
                className="rounded-2xl border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-4 flex flex-col justify-between gap-3 hover:border-cyan-500/40 transition-colors"
              >
                {/* Header */}
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="font-bold text-sm text-[var(--text-primary)]">
                        {r.drug_name} Rescue
                      </span>
                      <span className="font-mono-num text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 font-semibold">
                        {r.target_id} ({r.pdb_id}) vs {r.counter_target}
                      </span>
                    </div>
                    <p className="text-[11px] text-[var(--text-muted)] mt-0.5">{r.objective}</p>
                  </div>
                  <div className="flex items-center gap-1.5 font-mono-num text-[10px]">
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 font-semibold">
                      Core {r.core_preserved_ratio}
                    </span>
                    {r.posebusters_passed && (
                      <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 font-semibold">
                        PB PASS
                      </span>
                    )}
                  </div>
                </div>

                {/* Parent -> Rescued 2D RDKit SVGs */}
                <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 rounded-xl bg-[var(--bg-surface)] border border-[var(--border-subtle)] p-2.5">
                  <div className="flex flex-col items-center">
                    <span className="text-[9.5px] font-mono-num uppercase tracking-wider text-[var(--text-muted)] mb-1">
                      Clinical Parent ({r.drug_name})
                    </span>
                    <div
                      className="w-full max-w-[200px] h-[112px] flex items-center justify-center [&>svg]:max-h-full [&>svg]:max-w-full"
                      dangerouslySetInnerHTML={{ __html: parentSvg }}
                    />
                  </div>
                  <div className="flex flex-col items-center gap-1 text-emerald-500">
                    <ArrowRight className="h-4 w-4" />
                    <span className="text-[9px] font-mono-num font-bold uppercase">SAFE</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-[9.5px] font-mono-num uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-semibold mb-1">
                      SAFE-GPT Lead
                    </span>
                    <div
                      className="w-full max-w-[200px] h-[112px] flex items-center justify-center [&>svg]:max-h-full [&>svg]:max-w-full"
                      dangerouslySetInnerHTML={{ __html: analogSvg }}
                    />
                  </div>
                </div>

                {/* Before -> After KPIs */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono-num text-[11px]">
                  <div className="rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] px-2.5 py-1.5">
                    <div className="text-[9.5px] text-[var(--text-muted)]">On-Target ΔG</div>
                    <div className="font-bold text-cyan-600 dark:text-cyan-400">
                      {r.on_vina_dg.toFixed(2)}
                    </div>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] px-2.5 py-1.5">
                    <div className="text-[9.5px] text-[var(--text-muted)]">
                      ΔΔG vs {r.counter_target}
                    </div>
                    <div className="font-bold text-emerald-600 dark:text-emerald-400">
                      {r.ddg.toFixed(2)} kcal
                    </div>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] px-2.5 py-1.5">
                    <div className="text-[9.5px] text-[var(--text-muted)]">
                      {m3Drop > hergDrop && r.parent_chrm3 > 0.5 ? 'P(M3 Off-Tgt)' : 'P(hERG)'}
                    </div>
                    <div className="font-bold text-emerald-600 dark:text-emerald-400">
                      {m3Drop > hergDrop && r.parent_chrm3 > 0.5
                        ? `${r.parent_chrm3.toFixed(2)}→${r.analog_chrm3.toFixed(2)}`
                        : `${r.parent_herg.toFixed(2)}→${r.analog_herg.toFixed(2)}`}
                    </div>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] px-2.5 py-1.5">
                    <div className="text-[9.5px] text-[var(--text-muted)]">CNS MPO v2</div>
                    <div className="font-bold text-[var(--text-primary)]">
                      {r.parent_mpo.toFixed(2)}→
                      <span className="text-emerald-500">{r.analog_mpo.toFixed(2)}</span>
                    </div>
                  </div>
                </div>

                {/* Footer Win + 3D Inspect CTA */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[var(--border-subtle)]">
                  <span className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                    {r.headline_win}
                  </span>
                  <button
                    type="button"
                    onClick={() => onInspectTarget(r.target_id)}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25 cursor-pointer"
                  >
                    <Atom className="h-3.5 w-3.5" /> Inspect {r.target_id} 3D
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* PANEL C: 12-Endpoint Avoid-ome & DMPK + External Rat Kp,uu + Credibility Dossier */}
      <section className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        <div className="xl:col-span-8 lab-card rounded-2xl p-5 space-y-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-1.5 text-[10px] font-mono-num uppercase tracking-wider text-violet-600 dark:text-violet-400 font-bold">
                <ShieldCheck className="h-3.5 w-3.5" /> v4.0 Panel C · 12-Endpoint Industrial Avoid-ome &amp; DMPK Suite
              </div>
              <h3 className="font-bold text-base text-[var(--text-primary)] mt-0.5">
                5-Fold Bemis-Murcko Scaffold-CV ROC-AUC &amp; 90% Split-Conformal Coverage
              </h3>
              <p className="text-xs text-[var(--text-muted)]">
                Zero-leakage scaffold splits across BBB/efflux, HLM/CYP DMPK, cardiac/metabolic
                avoid-ome (5-HT2B, H1, M3), and AMES/DILI toxicology. Hover bars for ECE &amp;
                Split-Conformal UQ.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-[10.5px]">
              {(Object.entries(CATEGORY_COLORS) as [V4AvoidomeEndpoint['category'], string][]).map(
                ([cat, color]) => (
                  <span key={cat} className="inline-flex items-center gap-1 text-[var(--text-secondary)]">
                    <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
                    {cat}
                  </span>
                ),
              )}
            </div>
          </div>

          <div className="h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={v4.avoidome_12ep.endpoints}
                layout="vertical"
                margin={{ top: 4, right: 20, left: 10, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} horizontal={false} />
                <XAxis type="number" domain={[0.72, 1.0]} tick={{ fill: axisTick, fontSize: 10.5 }} />
                <YAxis
                  type="category"
                  dataKey="label"
                  width={132}
                  tick={{ fill: axisTick, fontSize: 10.5 }}
                />
                <ReferenceLine x={0.85} stroke="#10B981" strokeDasharray="4 4" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const ep = payload[0].payload as V4AvoidomeEndpoint;
                    return (
                      <div className="lab-card rounded-xl p-3 shadow-xl text-xs space-y-1 max-w-xs">
                        <div className="font-bold text-[var(--text-primary)]">
                          {ep.label}{' '}
                          <span className="text-[10px] font-mono-num text-cyan-500">
                            ({ep.category})
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono-num text-[11px]">
                          <span>
                            ROC-AUC:{' '}
                            <strong className="text-emerald-500">
                              {ep.roc_auc.toFixed(3)} ± {ep.roc_auc_std.toFixed(3)}
                            </strong>
                          </span>
                          <span>
                            Samples: <strong>{ep.n_samples.toLocaleString()}</strong>
                          </span>
                          <span>
                            Platt ECE: <strong>{ep.ece.toFixed(4)}</strong>
                          </span>
                          {ep.conformal_coverage_pct !== null && (
                            <span>
                              90% Conformal: <strong>{ep.conformal_coverage_pct.toFixed(1)}%</strong>
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="roc_auc" radius={[0, 6, 6, 0]}>
                  {v4.avoidome_12ep.endpoints.map((ep) => (
                    <Cell key={ep.key} fill={CATEGORY_COLORS[ep.category]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column: Rat Kp,uu External Validation + FDA/EMA Credibility Dossier */}
        <div className="xl:col-span-4 flex flex-col gap-5">
          <div className="lab-card rounded-2xl p-5 space-y-3 flex-1">
            <div className="flex items-center gap-1.5 text-[10px] font-mono-num uppercase tracking-wider text-cyan-600 dark:text-cyan-400 font-bold">
              <Activity className="h-3.5 w-3.5" /> External Quantitative Brain-Penetration Validation
            </div>
            <h3 className="font-bold text-base text-[var(--text-primary)]">
              Rat Unbound Brain/Plasma Ratio (K<sub>p,uu,brain</sub>, n={kpuu.n_compounds})
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              External non-overlapping in vivo unbound brain-to-plasma dataset verifying that
              GenEuclidean&apos;s calibrated BBB oracle tracks true free CNS exposure rather than
              total lipophilic binding.
            </p>
            <div className="grid grid-cols-2 gap-2.5 pt-1 font-mono-num">
              <div className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-3">
                <div className="text-[10px] text-[var(--text-muted)]">BBB Spearman ρ</div>
                <div className="text-xl font-extrabold text-emerald-500">
                  +{kpuu.bbb_spearman_rho.toFixed(3)}
                </div>
                <div className="text-[10px] text-[var(--text-muted)]">
                  p = {formatPValue(kpuu.bbb_spearman_p)}
                </div>
              </div>
              <div className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-3">
                <div className="text-[10px] text-[var(--text-muted)]">High-K_p,uu AUC</div>
                <div className="text-xl font-extrabold text-cyan-500">
                  {kpuu.bbb_roc_auc_high_kpuu.toFixed(3)}
                </div>
                <div className="text-[10px] text-[var(--text-muted)]">
                  MPO ρ = +{kpuu.mpo_spearman_rho.toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="lab-card rounded-2xl p-5 space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[10px] font-mono-num uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-bold flex items-center gap-1">
                <ShieldCheck className="h-3.5 w-3.5" /> FDA / EMA AI Credibility Dossier
              </span>
              <span className="font-mono-num text-xs font-extrabold px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-300">
                {dossier.fda_ema_principles_total} checklist items
              </span>
            </div>
            <div className="text-xs text-[var(--text-secondary)] space-y-1">
              <div>
                <strong>Risk Tier:</strong>{' '}
                <span className="font-mono-num text-[11px]">{dossier.risk_tier}</span>
              </div>
              <div>
                <strong>SHA-256 Tracked Artifacts:</strong>{' '}
                <span className="font-mono-num">{dossier.n_tracked_artifacts}</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)] leading-relaxed pt-1 border-t border-[var(--border-subtle)]">
                {dossier.boundary}
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* PANEL D: Tourette CSTC Basal-Ganglia & GABAergic 4-Pocket Validation */}
      <section className="lab-card rounded-2xl p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono-num uppercase tracking-wider text-amber-600 dark:text-amber-400 font-bold">
              <Brain className="h-3.5 w-3.5" /> v4.0 Panel D · Tourette CSTC Basal-Ganglia &amp; GABAergic Circuit Expansion
            </div>
            <h3 className="font-bold text-base sm:text-lg text-[var(--text-primary)] mt-0.5">
              4 New Inhibitory Interneuron &amp; Indirect-Pathway Pockets (GABRA5, GAT1, ADORA2A, GRM5)
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              Targets the root neurobiology of Tourette Syndrome — striatal GABA interneuron
              disinhibition and CSTC indirect-pathway excitability — with zero-shot held-out 3D
              docking and Gate-1 validation.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 font-mono-num text-[11px]">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 border border-emerald-500/25 font-semibold">
              Top-1 RMSD: {v4.ts_circuit_gaba.median_top1_rmsd.toFixed(2)} Å
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/25 font-semibold">
              Anchor: {v4.ts_circuit_gaba.mean_anchor_pct.toFixed(0)}% · PB:{' '}
              {v4.ts_circuit_gaba.mean_pb_pct.toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-center">
          {/* Bar Chart: Binding Strength (-Vina dG) vs Clinical Ref & ZINC */}
          <div className="xl:col-span-5 h-[270px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tsGabaChartData} margin={{ top: 10, right: 10, left: -12, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
                <XAxis dataKey="target" tick={{ fill: axisTick, fontSize: 11 }} />
                <YAxis
                  domain={[0, 11]}
                  tick={{ fill: axisTick, fontSize: 10.5 }}
                  label={{
                    value: '−Vina ΔG (kcal/mol, ↑ stronger)',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 18,
                    fill: axisTick,
                    fontSize: 10,
                  }}
                />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(v: number, name: string) => [`−${v.toFixed(2)} kcal/mol`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '10.5px' }} />
                <Bar
                  dataKey="geneuclidean_strength"
                  name="GenEuclidean Lead"
                  fill="#10B981"
                  radius={[5, 5, 0, 0]}
                />
                <Bar
                  dataKey="clinical_strength"
                  name="Clinical Reference"
                  fill="#38BDF8"
                  radius={[5, 5, 0, 0]}
                />
                <Bar
                  dataKey="zinc_strength"
                  name="ZINC Baseline"
                  fill="#64748B"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 4 Pocket Cards with 1-Click 3D Studio Launch */}
          <div className="xl:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {v4.ts_circuit_gaba.pockets.map((p) => (
              <div
                key={p.target}
                className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-3.5 flex flex-col justify-between gap-2.5 hover:border-amber-500/40 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono-num font-extrabold text-sm text-[var(--text-primary)]">
                        {p.target}
                      </span>
                      <span className="font-mono-num text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-300">
                        {p.pdb_id}
                      </span>
                    </div>
                    <div className="text-[11px] text-[var(--text-muted)] line-clamp-1">
                      {p.cstc_node} · {p.polarity}
                    </div>
                  </div>
                  <span className="font-mono-num text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 font-semibold shrink-0">
                    RMSD {p.gate1_top1_rmsd.toFixed(2)} Å
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-1.5 font-mono-num text-[10.5px]">
                  <div className="rounded-lg bg-[var(--bg-surface)] px-2 py-1 border border-[var(--border-subtle)]">
                    <div className="text-[9px] text-[var(--text-muted)]">Vina ΔG</div>
                    <div className="font-bold text-emerald-500">{p.best_vina_dg.toFixed(2)}</div>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-surface)] px-2 py-1 border border-[var(--border-subtle)]">
                    <div className="text-[9px] text-[var(--text-muted)]">CNS MPO</div>
                    <div className="font-bold text-cyan-500">{p.best_mpo.toFixed(2)}</div>
                  </div>
                  <div className="rounded-lg bg-[var(--bg-surface)] px-2 py-1 border border-[var(--border-subtle)]">
                    <div className="text-[9px] text-[var(--text-muted)]">P(hERG)</div>
                    <div className="font-bold text-[var(--text-primary)]">
                      {(p.best_herg * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2 pt-1 border-t border-[var(--border-subtle)]">
                  <span className="text-[10.5px] text-[var(--text-muted)] truncate" title={p.clinical_ref_note}>
                    {p.clinical_ref_note}
                  </span>
                  <button
                    type="button"
                    onClick={() => onInspectTarget(p.target)}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[10.5px] font-semibold bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30 hover:bg-amber-500/25 shrink-0 cursor-pointer"
                  >
                    <Atom className="h-3 w-3" /> 3D Complex
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
