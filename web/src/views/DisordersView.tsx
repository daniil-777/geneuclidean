import { ArrowRight, Atom, Brain, CheckCircle2, Sparkles } from 'lucide-react';
import type { MoleculeIndexEntry, V4Discovery } from '../types/studio';

interface DisordersViewProps {
  molecules: MoleculeIndexEntry[];
  v4Discovery: V4Discovery;
  onLaunchStudio: (target: string, disorder: string) => void;
}

interface PortalSpec {
  id: string;
  name: string;
  subtitle: string;
  accent: string;
  borderAccent: string;
  clinicalContext: string;
  therapeuticGoal: string;
  benchmarkDrugs: string[];
  v4Slot?: boolean;
}

const PORTALS: PortalSpec[] = [
  {
    id: 'Tourette Syndrome',
    name: 'Tourette Syndrome & Tic Disorders',
    subtitle: 'Direct-Pathway D1 Anti-EPS, Presynaptic VMAT2 & Striatal Gating',
    accent: 'from-cyan-500/18 to-emerald-500/10',
    borderAccent: 'border-cyan-500/40',
    clinicalContext:
      'Inspired by Emalex/Teva Phase 3 D1AMOND ecopipam (JAMA Neurology 2026) and VMAT2 inhibitors (valbenazine), avoiding D2-receptor extrapyramidal symptoms (EPS) and tardive dyskinesia.',
    therapeuticGoal:
      'Selective DRD1 antagonism (ΔΔG_D1−D2 < −1.0 kcal/mol), reversible VMAT2 modulation, genetic HDC/HRH3 rescue, and pediatric prefrontal α2A support.',
    benchmarkDrugs: ['Ecopipam (Ph3)', 'Valbenazine', 'Guanfacine ER', 'Pitolisant'],
  },
  {
    id: 'Schizophrenia',
    name: 'Schizophrenia & Non-D2 Antipsychotics',
    subtitle: 'Muscarinic M4, Presynaptic TAAR1, Striatal PDE10A & 5-HT2A',
    accent: 'from-violet-500/18 to-sky-500/10',
    borderAccent: 'border-violet-500/40',
    clinicalContext:
      'Following BMS/Karuna Cobenfy (FDA Sep 2024, CHRM4/CHRM1) and TAAR1/PDE10A programs targeting positive, negative, and cognitive symptoms without classical D2 motor liabilities.',
    therapeuticGoal:
      'High CNS MPO (>4.5), low hERG QT liability (<20%), and orthosteric/allosteric modulation across CHRM4, TAAR1, PDE10A, DRD3, and HTR2A.',
    benchmarkDrugs: ['Cobenfy (Xanomeline)', 'Ulotaront', 'Lumateperone', 'Cariprazine'],
  },
  {
    id: 'OCD / ADHD',
    name: 'Comorbid OCD, ADHD & Affective Circuits',
    subtitle: 'Cortico-Striato-Thalamo-Cortical SERT, DAT & Prefrontal α2A',
    accent: 'from-emerald-500/18 to-teal-500/10',
    borderAccent: 'border-emerald-500/40',
    clinicalContext:
      'Over 60% of Tourette Syndrome patients present with comorbid OCD or ADHD. Clean monoamine transporter and adrenergic pharmacology addresses overlapping executive and compulsive circuits.',
    therapeuticGoal:
      'High brain penetration P(BBB+) > 0.85, minimal P-gp efflux (<10%), and sub-nanomolar ligand efficiency across SERT, DAT, and ADRA2A.',
    benchmarkDrugs: ['Escitalopram', 'Paroxetine', 'Dexmethylphenidate', 'Guanfacine'],
  },
  {
    id: "Alzheimer's Disease",
    name: "Alzheimer's Disease & Neurodegeneration",
    subtitle: 'Cholinergic/Histaminergic Cognition + v4.0 Disease-Modifying Expansion',
    accent: 'from-amber-500/18 to-rose-500/10',
    borderAccent: 'border-amber-500/40',
    clinicalContext:
      'Bridges active v3.1 cognitive & dementia-agitation targets (CHRM4, HRH3, HTR2A — cf. pimavanserin/xanomeline) with GenEuclidean v4.0 geriatric-safe amyloid, tau, and neuroinflammation pipelines.',
    therapeuticGoal:
      'Strict geriatric CNS safety (low hERG/CYP/P-gp, high Kp,uu) across cholinergic M4/H3/5-HT2A and upcoming v4.0 BACE1, GSK3β, OGA, NLRP3, and RIPK1 pockets.',
    benchmarkDrugs: ['Xanomeline', 'Pimavanserin', 'Donepezil', 'v4.0 Disease-Modifying'],
    v4Slot: true,
  },
];

export function DisordersView({
  molecules,
  v4Discovery,
  onLaunchStudio,
}: DisordersViewProps) {
  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6 sm:py-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[11px] font-mono-num uppercase tracking-widest text-cyan-600 dark:text-cyan-400 font-semibold">
            Translational CNS Portals
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-primary)]">
            Neurological &amp; Psychiatric Discovery Portals
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
            Curated clinical indications linking human structural biology, mechanistic rationale, and
            verified <em>in silico</em> de novo leads directly into the 3D Molecular Studio.
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl lab-card text-xs">
          <Brain className="h-4 w-4 text-violet-500" />
          <span className="text-[var(--text-secondary)]">
            Auto-extensible for{' '}
            <strong className="font-mono-num text-[var(--text-primary)]">v4.0</strong> targets
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {PORTALS.map((portal) => {
          const matched = molecules.filter((m) => m.disorders.includes(portal.id));
          return (
            <div
              key={portal.id}
              className={`rounded-2xl border ${portal.borderAccent} bg-gradient-to-br ${portal.accent} p-5 flex flex-col justify-between gap-4`}
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <h2 className="text-lg font-bold text-[var(--text-primary)]">{portal.name}</h2>
                      {portal.v4Slot && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-num font-bold bg-amber-500/20 text-amber-700 dark:text-amber-300 border border-amber-500/35">
                          <Sparkles className="h-3 w-3" />
                          {v4Discovery.detected ? 'v4.0 Live' : 'v4.0 Ready'}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-cyan-700 dark:text-cyan-300 font-medium mt-0.5">
                      {portal.subtitle}
                    </p>
                  </div>
                  <span className="font-mono-num text-xs font-bold px-2.5 py-1 rounded-lg lab-card shrink-0">
                    {matched.length} active
                  </span>
                </div>

                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                  {portal.clinicalContext}
                </p>

                <div className="rounded-xl bg-[var(--bg-surface)]/80 border border-[var(--border-subtle)] p-3 text-xs text-[var(--text-secondary)]">
                  <span className="font-semibold text-[var(--text-primary)]">Design Objective: </span>
                  {portal.therapeuticGoal}
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10.5px] uppercase tracking-wider font-semibold text-[var(--text-muted)] mr-1">
                    Clinical Benchmarks:
                  </span>
                  {portal.benchmarkDrugs.map((drug) => (
                    <span
                      key={drug}
                      className="px-2 py-0.5 rounded-md text-[11px] bg-[var(--bg-surface)]/85 border border-[var(--border-subtle)] text-[var(--text-primary)] font-medium"
                    >
                      {drug}
                    </span>
                  ))}
                </div>

                {/* Active target lead pills linking straight to 3D Studio */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                  {matched.map((mol) => (
                    <button
                      key={`${portal.id}-${mol.target}`}
                      type="button"
                      onClick={() => onLaunchStudio(mol.target, portal.id)}
                      className="group text-left rounded-xl p-2.5 bg-[var(--bg-surface)]/90 border border-[var(--border-subtle)] hover:border-cyan-500/60 transition-all cursor-pointer"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-bold text-xs text-[var(--text-primary)] group-hover:text-cyan-500 transition-colors">
                          {mol.lead_id} ({mol.pdb_id})
                        </span>
                        <span className="font-mono-num text-[10.5px] font-bold text-emerald-600 dark:text-emerald-400">
                          {mol.vina_dg.toFixed(2)} kcal
                        </span>
                      </div>
                      <div className="flex items-center justify-between gap-2 mt-1 font-mono-num text-[10px] text-[var(--text-muted)]">
                        <span>MPO {mol.cns_mpo_v2.toFixed(2)}</span>
                        <span>hERG {(mol.herg_prob * 100).toFixed(0)}%</span>
                        <span className="text-cyan-600 dark:text-cyan-400 flex items-center gap-0.5 font-sans font-semibold">
                          3D <ArrowRight className="h-3 w-3" />
                        </span>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Auto-discovery slot for v4.0 Alzheimer's targets */}
                {portal.v4Slot && (
                  <div className="rounded-xl border border-dashed border-amber-500/40 bg-[var(--bg-surface)]/65 p-3 space-y-2">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-bold text-amber-700 dark:text-amber-300">
                        v4.0 Auto-Discovery Target Slots (Alzheimer&apos;s Pipeline)
                      </span>
                      <span className="font-mono-num text-[10px] text-[var(--text-muted)]">
                        results/tables/v4_*.json
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1.5">
                      {v4Discovery.roadmap_targets.map((rt) => (
                        <div
                          key={rt.target}
                          className="rounded-lg px-2 py-1.5 bg-[var(--bg-elevated)]/80 border border-[var(--border-subtle)] text-[10.5px]"
                        >
                          <div className="font-mono-num font-bold text-[var(--text-primary)]">
                            {rt.target}
                          </div>
                          <div className="text-[var(--text-muted)] truncate">{rt.mechanism}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between">
                <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  100% PoseBusters · Strict ChEMBL Novelty
                </span>
                {matched[0] && (
                  <button
                    type="button"
                    onClick={() => onLaunchStudio(matched[0].target, portal.id)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500/18 text-cyan-700 dark:text-cyan-200 border border-cyan-500/35 hover:bg-cyan-500/28 cursor-pointer"
                  >
                    <Atom className="h-3.5 w-3.5" /> Open Portal in 3D Studio
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
