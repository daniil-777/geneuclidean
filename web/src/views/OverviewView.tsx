import { ArrowRight, ArrowUpRight, Atom, BookOpen, ChartNoAxesCombined, ChevronDown, ChevronRight, Fingerprint, Orbit, ScanLine, Workflow } from 'lucide-react';
import { lazy, Suspense, useEffect, useState } from 'react';
import structureImage from '../assets/drd1-study.webp';
import { loadComplexDetail } from '../lib/data';
import type { ComplexDetail, OverviewData, StudioTab, ThemeMode } from '../types/studio';

const Mol3DViewer = lazy(() => import('../components/Mol3DViewer').then(module => ({ default: module.Mol3DViewer })));

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
    title: 'Chiral PaiNN-SE3 3D Encoder',
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
    title: 'PocketPerceiverResampler Bridge',
    summary: 'Permutation- & SE(3)-invariant compression into 24 scale-matched soft prefix tokens.',
    details: [
      '2-layer 8-head cross-attention extracts 16 contact-focused queries + 4 PocketCVAE + 4 global/pKd tokens.',
      'Exact 0.0 attention on padded atoms and ScaleMatchRMSNorm eliminating Step-0 activation shock.',
      'Achieves 7.12× <4.5 Å crystal contact enrichment and 4.25× canonical CNS hotspot enrichment.',
    ],
  },
  {
    id: 'backbone',
    title: '3 Local LLMs + 1.79 MB Browser AI',
    summary:
      'Multi-backbone chemical foundation transformers enabling de novo generation AND 3D scaffold rescue.',
    details: [
      'SAFE-GPT-87M (fragment SAFE + 3D clinical scaffold decoration), NovoMolGen-157M (MPS SDPA, IntDiv1=0.892), and GP-MoLFormer-Uniq.',
      '12.5× exact fast_4d_tril causal attention kernel + 11.15× KV-cached Classifier-Free Guidance (w=1.3).',
      'Distilled into a 1.79 MB INT8 KV-cached client-side Transformer running in the browser generator.',
    ],
  },
  {
    id: 'alignment',
    title: '4-Ch GDPO + L_XP-DPO + 12-Ep Avoid-ome',
    summary:
      'Multiplicative safety-gated alignment across 12 industrial DMPK/Avoid-ome oracles & Split-Conformal UQ.',
    details: [
      'SAFE-GPT-87M + 4-Ch GDPO lifts 12-endpoint Pharma-Safe 5.5× vs v3.1 (6.8% → 37.3%, 22/24 Holm) with 0.0% DCU and 0.0% PDE10A sink.',
      'Slashes Biogen MDR1-MDCK efflux 7.6× (0.448 → 0.059) and validates external Rat Kp,uu (ρ = +0.724, p = 3.85×10⁻⁴⁰).',
      'The campaign evaluated generated poses with held-out AutoDock Vina and recorded its PoseBusters results in a credibility dossier.',
    ],
  },
] as const;

const WORKSPACES = [
  { id: 'studio3d', label: '3D viewer', description: 'Inspect docked poses and the interactions inside a binding pocket.', icon: Orbit, shade: 'violet' },
  { id: 'generator', label: 'Molecule generator', description: 'Generate candidate structures around a target with an in-browser model.', icon: Fingerprint, shade: 'mint' },
  { id: 'circuit', label: 'Circuit builder', description: 'Connect encoders, decoders and scoring functions into a pipeline.', icon: Workflow, shade: 'blue' },
] as const;

export function OverviewView({ overview, theme, onNavigate }: OverviewViewProps) {
  const [viewerOpen, setViewerOpen] = useState(false);
  const [target, setTarget] = useState<string>('DRD1');
  const [complex, setComplex] = useState<ComplexDetail | null>(null);
  const [viewerError, setViewerError] = useState(false);
  const [activeStage, setActiveStage] = useState<string>('encoder');

  useEffect(() => {
    if (!viewerOpen) return;
    let alive = true;
    setViewerError(false);
    setComplex(null);
    loadComplexDetail(target).then(data => { if (alive) setComplex(data); })
      .catch(() => { if (alive) setViewerError(true); });
    return () => { alive = false; };
  }, [viewerOpen, target]);

  const stage = PIPELINE_STAGES.find(item => item.id === activeStage) ?? PIPELINE_STAGES[0];

  return (
    <div className="studio-overview">
      <section className="studio-hero" aria-labelledby="studio-hero-title">
        <div className="studio-hero-copy">
          <span className="studio-feature-label"><ScanLine size={16} aria-hidden="true"/> Tourette study</span>
          <h1 id="studio-hero-title">Molecular<br/>discovery studio</h1>
          <p>Explore drug candidates in 3D and simulate how a Tourette research compound moves from the gut to the brain.</p>
          <a className="studio-hero-action" href="#human-twin">Open simulator <ArrowRight size={18}/></a>
        </div>
        <figure className="studio-hero-figure">
          <div className="studio-scientific-plate">
            <img className="studio-render" src={structureImage} width="1320" height="950" fetchPriority="high"
              alt="Ball-and-stick rendering of the docked Tourette research candidate DRD1_v53_L1."/>
            <i className="studio-crosshair first" aria-hidden="true"/><i className="studio-crosshair last" aria-hidden="true"/>
          </div>
          <figcaption><span className="studio-caption-dot" aria-hidden="true"/><strong>DRD1_v53_L1</strong><span>Docked conformation · 9LLG</span></figcaption>
        </figure>
      </section>

      <section className="studio-workspaces" aria-labelledby="workspaces-title">
        <div className="studio-section-heading"><h2 id="workspaces-title">Research workspaces</h2>
          <a href="#experiments">View experiments <ArrowUpRight size={16}/></a></div>
        <div className="studio-workspace-grid">
          {WORKSPACES.map(({ id, label, description, icon: Icon, shade }) => (
            <a href={`#${id}`} key={id} className={`studio-workspace ${shade}`}>
              <div className="studio-workspace-top"><span className="studio-tool-icon"><Icon size={24} strokeWidth={1.7}/></span><ArrowUpRight size={19}/></div>
              <h3>{label}</h3><p>{description}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="studio-reference" aria-labelledby="studio-reference-title">
        <div className="studio-section-heading"><h2 id="studio-reference-title">Inside the studio</h2>
          <a href="#figures">Figures &amp; docs <ArrowUpRight size={16}/></a></div>
        <details onToggle={e => setViewerOpen(e.currentTarget.open)}>
          <summary><Atom size={19}/><span>Binding pocket preview</span><ChevronDown size={18}/></summary>
          {viewerOpen && <div className="studio-details-content">
            <div className="studio-pocket-controls">
              <label>Protein target <select value={target} onChange={e => setTarget(e.target.value)}>
                {HERO_TARGETS.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select></label>
              <button type="button" className="studio-inline-action" onClick={() => onNavigate('studio3d',target)}>Inspect in 3D viewer <ArrowRight size={14}/></button>
            </div>
            {viewerError ? <p role="status">This structure could not be loaded. Select a target to try again.</p> : complex
              ? <Suspense fallback={<p role="status">Loading structure viewer…</p>}><Mol3DViewer complex={complex} appTheme={theme} compact className="h-[350px] w-full"/></Suspense>
              : <p role="status">Loading binding pocket…</p>}
          </div>}
        </details>
        <details>
          <summary><Workflow size={19}/><span>Model architecture</span><ChevronDown size={18}/></summary>
          <div className="studio-details-content">
            <div className="studio-stage-tabs" aria-label="Pipeline stages">
              {PIPELINE_STAGES.map(item => <button type="button" key={item.id} onClick={() => setActiveStage(item.id)}
                aria-pressed={stage.id === item.id}>{item.id}</button>)}
            </div>
            <h3>{stage.title}</h3><p>{stage.summary}</p>
            <ul>{stage.details.map(line => <li key={line}>{line}</li>)}</ul>
          </div>
        </details>
        <details>
          <summary><BookOpen size={19}/><span>Release history</span><ChevronDown size={18}/></summary>
          <div className="studio-details-content studio-history">
            {overview.evolution.map(item => <article key={item.version}>
              <span>{item.version} · {item.year}</span><h3>{item.title}</h3>
              <p>{item.architecture}</p><ul>{item.highlights.map(line => <li key={line}>{line}</li>)}</ul>
            </article>)}
          </div>
        </details>
        <a className="studio-reference-link" href="#disorders"><ChartNoAxesCombined size={19}/><span>Browse disease targets</span><ChevronRight size={18}/></a>
      </section>
    </div>
  );
}
