import {
  BookOpen,
  Check,
  Copy,
  ExternalLink,
  FileText,
  Maximize2,
  Terminal,
  X,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { copyToClipboard, figureUrl } from '../lib/data';
import type { FigureItem } from '../types/studio';

interface FiguresViewProps {
  figures: FigureItem[];
  repoUrl: string;
}

const MANUSCRIPTS = [
  {
    title: 'v4.0 Tri-Disorder 3-LLM, Clinical Rescue & Avoid-ome Technical Report',
    path: 'docs/v4_tri_disorder_report.md',
    desc: '28 human targets (Tourette-GABA × Schizophrenia × Alzheimer’s), SAFE-GPT-87M vs NovoMolGen-157M vs GP-MoLFormer, 12-ep Avoid-ome + Rat Kp,uu, and FDA/EMA dossier.',
    badge: 'v4.0 Flagship',
  },
  {
    title: 'Tourette CSTC Basal-Ganglia & GABAergic Circuit Audit',
    path: 'docs/tourette_circuit_audit.md',
    desc: 'Independent neurobiology audit of 13 CSTC targets, 6 caught structural/species traps, and zero-shot 3D transfer onto GABRA5, GAT1, ADORA2A & GRM5.',
    badge: 'TS-GABA Circuit',
  },
  {
    title: 'GenEuclidean Python Library SDK Documentation',
    path: 'docs/library_api.md',
    desc: 'Clean lazy-import Python SDK (`import geneuclidean as ge`) covering Pocket loading, 3D generation, MedChem triage, docking, and 3D visualization.',
    badge: 'Python SDK',
  },
  {
    title: 'v3.1 CNS Neuroscience & Tourette Computational Chemistry Manuscript',
    path: 'docs/neuro_tourette_psychiatry.md',
    desc: 'Complete 4-gate, 3-iteration closed-loop CNS study, Basic-Amine Paradox resolution, Ablations A1–A5, 10×10 selectivity, and Top-12 dossiers.',
    badge: 'v3.1 Manuscript',
  },
  {
    title: '2024–2026 Industry Pharma Dataset Architecture & Data Hygiene',
    path: 'docs/industry_datasets.md',
    desc: 'PLINDER-2026, HiQBind, SAIR, MISATO, GPCRdb, B3DB, TDC, Biogen/Polaris, OpenADMET & 7 empirically caught data-hygiene traps.',
    badge: 'Industry Data',
  },
  {
    title: 'Chemical Foundation LLM Research & SE(3) Architecture Specs',
    path: 'docs/architecture.md',
    desc: 'Chiral PaiNN-SE3 equations, PocketPerceiverResampler, ScaleMatchRMSNorm, SAFE codec, and CPU/MPS numerical equivariance proofs.',
    badge: 'SE(3) Math',
  },
];

const CLI_SNIPPETS = [
  {
    title: 'Python Library SDK Quickstart (<200 ms lazy import)',
    cmd: `import geneuclidean as ge
pocket = ge.Pocket.from_target("DRD1")
scores = ge.score_molecules(["CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"])
ge.visualize_complex(pocket.pdb_path, "hit.sdf", out_html="studio.html")`,
  },
  {
    title: 'Inspect 28 Human Tri-Disorder & Tourette CSTC/GABA Targets',
    cmd: '.venv/bin/geneuclidean cns-targets --target DRD1',
  },
  {
    title: 'Score Any Molecule (Pfizer CNS MPO v2 + 12-Ep Avoid-ome & ADMET)',
    cmd: '.venv/bin/geneuclidean cns-admet "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"',
  },
  {
    title: 'Render Interactive 3Dmol.js WebGL Studio + 300-DPI Complex PNG',
    cmd: '.venv/bin/geneuclidean viz3d --pocket pocket.pdb --ligand hit.sdf --out-html complex.html --out-png complex.png --theme dark',
  },
  {
    title: 'Rebuild Web Studio JSON Data Assets (28 Complexes + v4 Benchmarks)',
    cmd: '.venv/bin/python scripts/build_web_data.py',
  },
];

const BIBTEX = `@software{emtsev2026geneuclidean,
  author       = {Emtsev, Daniil},
  title        = {{GenEuclidean v4.0: SE(3)-Equivariant 3D Pocket-Conditioned Chemical Foundation LLMs, Industrial Avoid-ome Alignment, and Live Browser AI Molecular Studio}},
  year         = {2026},
  version      = {4.0.0},
  url          = {https://github.com/daniil-777/geneuclidean}
}`;

export function FiguresView({ figures, repoUrl }: FiguresViewProps) {
  const [category, setCategory] = useState<string>('All');
  const [lightbox, setLightbox] = useState<FigureItem | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  const categories = useMemo(
    () => ['All', ...Array.from(new Set(figures.map((f) => f.category)))],
    [figures],
  );

  const visibleFigures = useMemo(
    () => (category === 'All' ? figures : figures.filter((f) => f.category === category)),
    [figures, category],
  );

  const handleCopy = async (key: string, text: string) => {
    const ok = await copyToClipboard(text);
    if (ok) {
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 1500);
    }
  };

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6 sm:py-8 space-y-10">
      {/* Header & Category Filter */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <div className="text-[11px] font-mono-num uppercase tracking-widest text-cyan-600 dark:text-cyan-400 font-semibold">
              300-DPI Publication Gallery
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-[var(--text-primary)]">
              Figures, Manuscripts &amp; Reproducibility Hub
            </h1>
            <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1">
              All {figures.length} publication-grade 300-DPI figures rendered directly from verified
              evaluation tables — click any figure for high-resolution inspection.
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                  category === cat
                    ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                    : 'lab-card text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* High-res figure grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {visibleFigures.map((fig) => (
            <div
              key={fig.id}
              onClick={() => setLightbox(fig)}
              className="group lab-card rounded-2xl overflow-hidden cursor-pointer flex flex-col"
            >
              <div className="relative aspect-[16/10] bg-slate-900/50 overflow-hidden border-b border-[var(--border-subtle)]">
                <img
                  src={figureUrl(fig.file)}
                  alt={fig.title}
                  loading="lazy"
                  className="w-full h-full object-contain bg-white dark:bg-[#0B1120] group-hover:scale-[1.02] transition-transform duration-300"
                />
                <div className="absolute top-2.5 right-2.5 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg bg-slate-950/80 text-white backdrop-blur-md">
                  <Maximize2 className="h-3.5 w-3.5" />
                </div>
                <span className="absolute top-2.5 left-2.5 px-2 py-0.5 rounded-md text-[10px] font-mono-num font-bold bg-slate-950/80 text-cyan-300 backdrop-blur-md">
                  Fig {fig.id}
                </span>
              </div>

              <div className="p-4 flex-1 flex flex-col justify-between gap-2">
                <div className="space-y-1">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
                    {fig.category}
                  </div>
                  <h3 className="font-bold text-sm text-[var(--text-primary)] leading-snug">
                    {fig.title}
                  </h3>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed line-clamp-3">
                    {fig.caption}
                  </p>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-[var(--border-subtle)] font-mono-num text-[10px] text-[var(--text-muted)]">
                  <span>{fig.file}</span>
                  <span>{fig.size_kb.toFixed(0)} KB · 300 DPI</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Repository Technical Manuscripts */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-cyan-500" />
          <h2 className="text-xl font-bold text-[var(--text-primary)]">
            Technical Manuscripts &amp; Architecture Documentation
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {MANUSCRIPTS.map((doc) => (
            <a
              key={doc.path}
              href={`${repoUrl}/blob/main/${doc.path}`}
              target="_blank"
              rel="noreferrer"
              className="lab-card rounded-2xl p-4 flex flex-col justify-between gap-3 group hover:border-cyan-500/50"
            >
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono-num text-[10px] font-bold px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300">
                    {doc.badge}
                  </span>
                  <ExternalLink className="h-3.5 w-3.5 text-[var(--text-muted)] group-hover:text-cyan-500" />
                </div>
                <h3 className="font-bold text-sm text-[var(--text-primary)] group-hover:text-cyan-500 transition-colors">
                  {doc.title}
                </h3>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{doc.desc}</p>
              </div>
              <div className="font-mono-num text-[11px] text-[var(--text-muted)] flex items-center gap-1.5">
                <FileText className="h-3.5 w-3.5" /> {doc.path}
              </div>
            </a>
          ))}
        </div>
      </section>

      {/* Reproducibility SDK, CLI + Colab + BibTeX */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 lab-card rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-emerald-500" />
              <h2 className="font-bold text-base text-[var(--text-primary)]">
                Python SDK, CLI &amp; Google Colab Commands
              </h2>
            </div>
            <a
              href={`${repoUrl}/blob/main/notebooks/GenEuclidean_v3_EndToEnd_Colab.ipynb`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs font-semibold text-cyan-600 dark:text-cyan-400 hover:underline"
            >
              Open Colab Notebook <ExternalLink className="h-3 w-3" />
            </a>
          </div>

          <div className="space-y-2.5">
            {CLI_SNIPPETS.map((snip) => (
              <div
                key={snip.title}
                className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-3 space-y-1.5"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-[var(--text-secondary)]">
                    {snip.title}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleCopy(snip.title, snip.cmd)}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10.5px] font-medium lab-card hover:border-cyan-500/50 text-[var(--text-primary)] cursor-pointer"
                  >
                    {copiedKey === snip.title ? (
                      <>
                        <Check className="h-3 w-3 text-emerald-500" /> Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" /> Copy
                      </>
                    )}
                  </button>
                </div>
                <pre className="font-mono-num text-[11px] text-cyan-700 dark:text-cyan-300 overflow-x-auto">
                  {snip.cmd}
                </pre>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-5 lab-card rounded-2xl p-5 flex flex-col justify-between gap-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-bold text-base text-[var(--text-primary)]">BibTeX Citation</h2>
              <button
                type="button"
                onClick={() => handleCopy('bibtex', BIBTEX)}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/35 hover:bg-cyan-500/25 cursor-pointer"
              >
                {copiedKey === 'bibtex' ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-500" /> Copied BibTeX
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" /> Copy BibTeX
                  </>
                )}
              </button>
            </div>
            <pre className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-3.5 font-mono-num text-[11px] text-[var(--text-secondary)] overflow-x-auto leading-relaxed">
              {BIBTEX}
            </pre>
          </div>

          <div className="rounded-xl bg-[var(--bg-elevated)]/70 border border-[var(--border-subtle)] p-3.5 text-xs text-[var(--text-secondary)] space-y-1">
            <div className="font-semibold text-[var(--text-primary)]">
              MIT License · Scaffold-split validation &amp; AI credibility dossier
            </div>
            <p>
              Verified on Apple Silicon M3 Pro (MPS) and Google Colab T4/L4/A100 GPUs across 28
              human 3D targets with strict Bemis-Murcko scaffold-disjoint validation.
            </p>
          </div>
        </div>
      </section>

      {/* Lightbox Modal */}
      {lightbox && (
        <div
          onClick={() => setLightbox(null)}
          className="fixed inset-0 z-50 bg-slate-950/88 backdrop-blur-md flex items-center justify-center p-3 sm:p-6"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="lab-card rounded-2xl max-w-6xl w-full max-h-[92vh] flex flex-col overflow-hidden shadow-2xl"
          >
            <div className="px-5 py-3.5 border-b border-[var(--border-subtle)] flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="font-bold text-sm sm:text-base text-[var(--text-primary)] truncate">
                  {lightbox.title}
                </div>
                <div className="text-xs text-[var(--text-muted)] truncate">{lightbox.file}</div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <a
                  href={figureUrl(lightbox.file)}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-semibold lab-card hover:border-cyan-500/50 text-[var(--text-primary)]"
                >
                  Open Raw PNG <ExternalLink className="h-3.5 w-3.5" />
                </a>
                <button
                  type="button"
                  onClick={() => setLightbox(null)}
                  className="p-1.5 rounded-lg lab-card hover:border-rose-500/50 text-[var(--text-secondary)] cursor-pointer"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="flex-1 overflow-auto bg-white dark:bg-[#0B1120] p-3 flex items-center justify-center">
              <img
                src={figureUrl(lightbox.file)}
                alt={lightbox.title}
                className="max-w-full max-h-[70vh] object-contain rounded-lg"
              />
            </div>

            <div className="px-5 py-3 border-t border-[var(--border-subtle)] text-xs text-[var(--text-secondary)] leading-relaxed">
              {lightbox.caption}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
