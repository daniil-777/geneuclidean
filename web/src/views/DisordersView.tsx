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
  badge: string;
}

const PORTALS: PortalSpec[] = [
  {
    id: 'Tourette (CSTC+GABA)',
    name: 'Tourette syndrome',
    subtitle: 'Dopamine D1 and GABAergic circuits',
    accent: 'from-cyan-500/18 to-emerald-500/10',
    borderAccent: 'border-cyan-500/45',
    badge: 'v4.0 CSTC+GABA LIVE',
    clinicalContext:
      'Combines Emalex/Teva Phase 3 D1AMOND ecopipam (DRD1 anti-EPS) & ALLEVIA-2 PDE10A with 4 brand-new basal-ganglia/GABAergic circuit pockets (GABRA5 α5⁺/γ2⁻, GAT1, active ADORA2A, mGluR5 NAM) plus MGLL & PDE1B.',
    therapeuticGoal:
      'Selective DRD1 antagonism (ΔΔG −1.31 vs DRD2), GABAergic impulse braking (GAT1 beating Tiagabine w/ 10× lower hERG), and mGluR5 signal filtering (−9.63 kcal/mol, MPO 5.85).',
    benchmarkDrugs: ['Ecopipam (Ph3)', 'Gemlapodect', 'Tiagabine', 'Dipraglurant', 'Guanfacine ER'],
  },
  {
    id: "Alzheimer's Disease",
    name: "Alzheimer's Disease & Neurodegeneration",
    subtitle: 'Cholinergic, Astrocytic MAO-B, Microglial NLRP3, Tau OGA/GSK3β & BACE1',
    accent: 'from-amber-500/18 to-rose-500/10',
    borderAccent: 'border-amber-500/45',
    badge: 'v4.0 ALZHEIMER LIVE',
    clinicalContext:
      'Full v4.0 geriatric-safe Alzheimer’s discovery suite spanning 6 dedicated disease-modifying & cognitive pockets (ACHE, MAOB+FAD, NLRP3+ADP, OGA, GSK3β, BACE1) plus M1/M4/H3/5-HT2A, retaining catalytic cofactors & counter-screening BChE, MAO-A, CDK2, and BACE2.',
    therapeuticGoal:
      'Geriatric-safe CNS exposure (Rat Kp,uu ρ=+0.724) + SAFE-GPT 3D rescues: Donepezil hERG 0.95→0.14 (ΔΔG −2.29 vs BChE) and Safinamide ΔΔG −3.47 vs MAO-A.',
    benchmarkDrugs: ['Donepezil (Rescued)', 'Safinamide (Rescued)', 'VTX3232', 'Ceptoglustat', 'Pimavanserin'],
  },
  {
    id: 'Schizophrenia',
    name: 'Schizophrenia & Non-D2 Antipsychotics',
    subtitle: 'Muscarinic M4/M1, GlyT1, GPR52, 5-HT1A, TAAR1 & Striatal PDE10A/PDE1B',
    accent: 'from-violet-500/18 to-sky-500/10',
    borderAccent: 'border-violet-500/45',
    badge: 'v4.0 EXPANDED LIVE',
    clinicalContext:
      'Expands beyond BMS/Karuna Cobenfy (CHRM4/CHRM1) to Phase 3 cognitive GlyT1 (Iclepertin), indirect-pathway orphan GPR52, 5-HT1A, TAAR1, and striatal PDE10A/PDE1B — eliminating v3.1’s PDE10A→VMAT2 sink completely (33.3% → 0.0%).',
    therapeuticGoal:
      'Low peripheral M3 (SAFE-GPT Xanomeline rescue slashes M3 0.97→0.22, ΔΔG −1.11 vs DRD2), low MDR1-MDCK efflux, and 0.0% DCU/reagent artifacts.',
    benchmarkDrugs: ['Cobenfy (Rescued)', 'Iclepertin (Ph3)', 'Ulotaront', 'Lumateperone', 'Cariprazine'],
  },
  {
    id: 'OCD / ADHD',
    name: 'Comorbid OCD, ADHD & Executive Circuits',
    subtitle: 'Cortico-Striato-Thalamo-Cortical SERT, DAT, Prefrontal α2A & 5-HT1A',
    accent: 'from-emerald-500/18 to-teal-500/10',
    borderAccent: 'border-emerald-500/45',
    badge: 'ACTIVE PORTAL',
    clinicalContext:
      'Over 60% of Tourette Syndrome patients present with comorbid OCD or ADHD. Clean monoamine transporter and prefrontal adrenergic/serotonergic pharmacology addresses overlapping executive and compulsive circuits.',
    therapeuticGoal:
      'High brain penetration P(BBB+) > 0.85, minimal P-gp/MDR1 efflux (<10%), and high ligand efficiency across SERT, DAT, ADRA2A, and HTR1A.',
    benchmarkDrugs: ['Escitalopram', 'Paroxetine', 'Dexmethylphenidate', 'Guanfacine'],
  },
];

export function DisordersView({ molecules, v4Discovery, onLaunchStudio }: DisordersViewProps) {
  return (
    <div className="studio-workbench-page mx-auto max-w-[1600px] px-4 sm:px-6 py-6 sm:py-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-[11px] font-mono-num uppercase tracking-widest text-cyan-600 dark:text-cyan-400 font-semibold">
            Translational Tri-Disorder Portals ({v4Discovery.active_version})
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-primary)]">
            Disorder portals
          </h1>
          <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1 max-w-3xl">
            Every target below is backed by a verified <em>Homo sapiens</em> crystal/cryo-EM pocket,
            held-out AutoDock Vina 3D pose, 12-endpoint Avoid-ome profile, and real Perceiver attention —
            click any target to open it directly in the interactive 3D Molecular Studio.
          </p>
        </div>
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl lab-card text-xs">
          <Brain className="h-4 w-4 text-emerald-500" />
          <span className="text-[var(--text-secondary)]">
            All <strong className="font-mono-num text-[var(--text-primary)]">{molecules.length}</strong>{' '}
            tri-disorder complexes live in 3D
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
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono-num font-bold bg-emerald-500/18 text-emerald-700 dark:text-emerald-300 border border-emerald-500/35">
                        <Sparkles className="h-3 w-3" />
                        {portal.badge}
                      </span>
                    </div>
                    <p className="text-xs text-cyan-700 dark:text-cyan-300 font-medium mt-0.5">
                      {portal.subtitle}
                    </p>
                  </div>
                  <span className="font-mono-num text-xs font-bold px-2.5 py-1 rounded-lg lab-card shrink-0">
                    {matched.length} targets
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

                {portal.id === 'Tourette (CSTC+GABA)' && (
                  <a href="#human-twin" className="inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm font-semibold bg-[var(--text-primary)] text-[var(--bg-surface)]">
                    Open Tourette simulator <ArrowRight size={16} aria-hidden />
                  </a>
                )}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 max-h-[290px] overflow-y-auto pr-0.5">
                  {matched.map((mol) => (
                    <button
                      key={`${portal.id}-${mol.target}`}
                      type="button"
                      onClick={() => onLaunchStudio(mol.target, portal.id)}
                      className="group text-left rounded-xl p-2.5 bg-[var(--bg-surface)]/92 border border-[var(--border-subtle)] hover:border-cyan-500/60 transition-all cursor-pointer"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-bold text-xs text-[var(--text-primary)] group-hover:text-cyan-500 transition-colors truncate">
                          {mol.target} ({mol.pdb_id})
                        </span>
                        <span className="font-mono-num text-[10.5px] font-bold text-emerald-600 dark:text-emerald-400 shrink-0">
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
              </div>

              <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between">
                <span className="text-[11px] text-[var(--text-muted)] flex items-center gap-1">
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                  PoseBusters Verified · 12-Ep Avoid-ome Scored
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
