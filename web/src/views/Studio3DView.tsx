import {
  AlertTriangle,
  Check,
  Copy,
  Crosshair,
  Download,
  ExternalLink,
  Maximize2,
  Search,
  ShieldCheck,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { INTERACTION_META, Mol3DViewer, type Mol3DViewerHandle } from '../components/Mol3DViewer';
import { copyToClipboard, downloadTextFile, loadComplexDetail } from '../lib/data';
import type { ComplexDetail, MoleculeIndexEntry, ThemeMode } from '../types/studio';

interface Studio3DViewProps {
  molecules: MoleculeIndexEntry[];
  initialTarget: string;
  initialDisorderFilter?: string;
  theme: ThemeMode;
  onSelectTarget: (target: string) => void;
}

const DISORDER_FILTERS = [
  'All',
  'Tourette Syndrome',
  'Schizophrenia',
  'OCD / ADHD',
  "Alzheimer's Disease",
] as const;

export function Studio3DView({
  molecules,
  initialTarget,
  initialDisorderFilter = 'All',
  theme,
  onSelectTarget,
}: Studio3DViewProps) {
  const viewerRef = useRef<Mol3DViewerHandle | null>(null);
  const [disorderFilter, setDisorderFilter] = useState<string>(initialDisorderFilter);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTarget, setSelectedTarget] = useState<string>(
    initialTarget || molecules[0]?.target || 'DRD1',
  );
  const [complex, setComplex] = useState<ComplexDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [copiedSmiles, setCopiedSmiles] = useState(false);
  const [zoom2D, setZoom2D] = useState(false);
  const [activeContactIdx, setActiveContactIdx] = useState<number | null>(null);

  useEffect(() => {
    if (initialTarget && initialTarget !== selectedTarget) {
      setSelectedTarget(initialTarget);
    }
  }, [initialTarget, selectedTarget]);

  useEffect(() => {
    if (initialDisorderFilter) {
      setDisorderFilter(initialDisorderFilter);
    }
  }, [initialDisorderFilter]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setActiveContactIdx(null);
    loadComplexDetail(selectedTarget)
      .then((data) => {
        if (alive) {
          setComplex(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [selectedTarget]);

  const filteredMolecules = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return molecules.filter((m) => {
      if (disorderFilter !== 'All' && !m.disorders.includes(disorderFilter)) {
        return false;
      }
      if (!q) return true;
      const hay = `${m.target} ${m.lead_id} ${m.pdb_id} ${m.title} ${m.disease_tag} ${m.canonical_smiles}`.toLowerCase();
      return hay.includes(q);
    });
  }, [molecules, disorderFilter, searchQuery]);

  const handleChooseTarget = (target: string) => {
    setSelectedTarget(target);
    onSelectTarget(target);
  };

  const handleCopySmiles = async () => {
    if (!complex) return;
    const ok = await copyToClipboard(complex.canonical_smiles);
    if (ok) {
      setCopiedSmiles(true);
      setTimeout(() => setCopiedSmiles(false), 1500);
    }
  };

  const handleDownloadSdf = () => {
    if (!complex || !complex.ligand_sdf) return;
    downloadTextFile(
      `${complex.lead_id}_${complex.pdb_id}_docked.sdf`,
      complex.ligand_sdf,
      'chemical/x-mdl-sdfile',
    );
  };

  return (
    <div className="mx-auto max-w-[1680px] px-3 sm:px-5 py-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
        {/* LEFT RAIL: Disorder Filters, Search & Target Cards */}
        <aside className="lg:col-span-3 xl:col-span-2.5 flex flex-col gap-3 lg:h-[calc(100vh-5.5rem)]">
          <div className="lab-card rounded-2xl p-3 space-y-2.5 shrink-0">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-600 dark:text-cyan-400">
                CNS Lead Portfolio
              </span>
              <span className="font-mono-num text-[11px] text-[var(--text-muted)]">
                {filteredMolecules.length}/{molecules.length}
              </span>
            </div>

            <div className="relative">
              <Search className="h-3.5 w-3.5 text-[var(--text-muted)] absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search target, PDB, SMILES…"
                className="w-full pl-8 pr-3 py-1.5 rounded-lg text-xs bg-[var(--bg-elevated)] border border-[var(--border-subtle)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-cyan-500/60"
              />
            </div>

            <div className="flex flex-wrap gap-1">
              {DISORDER_FILTERS.map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => setDisorderFilter(d)}
                  className={`px-2 py-0.5 rounded-md text-[10.5px] font-medium border transition-colors cursor-pointer ${
                    disorderFilter === d
                      ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-700 dark:text-cyan-200'
                      : 'border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                  }`}
                >
                  {d === "Alzheimer's Disease" ? "Alzheimer's" : d === 'Tourette Syndrome' ? 'Tourette' : d}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto space-y-2 pr-0.5 max-h-[340px] lg:max-h-none">
            {filteredMolecules.map((mol) => {
              const active = mol.target === selectedTarget;
              return (
                <button
                  key={mol.target}
                  type="button"
                  onClick={() => handleChooseTarget(mol.target)}
                  className={`w-full text-left rounded-xl p-2.5 border transition-all cursor-pointer ${
                    active
                      ? 'bg-gradient-to-br from-cyan-500/16 to-emerald-500/10 border-cyan-500/60 shadow-sm'
                      : 'lab-card hover:border-cyan-500/35'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="font-bold text-xs text-[var(--text-primary)] truncate">
                        {mol.lead_id}
                      </span>
                      {mol.is_curated_rank2 && (
                        <span className="text-[9px] font-mono-num px-1 rounded bg-sky-500/15 text-sky-600 dark:text-sky-300 border border-sky-500/25">
                          #2
                        </span>
                      )}
                    </div>
                    <span className="font-mono-num text-[10px] font-bold px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 shrink-0">
                      {mol.pdb_id}
                    </span>
                  </div>

                  <div className="text-[10.5px] text-[var(--text-muted)] truncate mt-0.5">
                    {mol.disease_tag}
                  </div>

                  {/* Mini 2D RDKit vector preview */}
                  <div
                    className="my-1.5 h-14 rounded-lg bg-[var(--bg-elevated)]/70 border border-[var(--border-subtle)] flex items-center justify-center overflow-hidden px-1 rdkit-svg-wrap"
                    dangerouslySetInnerHTML={{
                      __html: theme === 'dark' ? mol.svg_mini_dark : mol.svg_mini_light,
                    }}
                  />

                  <div className="flex items-center justify-between gap-1 font-mono-num text-[10px]">
                    <span className="px-1.5 py-0.5 rounded bg-emerald-500/12 text-emerald-700 dark:text-emerald-300 font-semibold">
                      {mol.vina_dg.toFixed(2)} kcal
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-slate-500/12 text-[var(--text-secondary)]">
                      MPO {mol.cns_mpo_v2.toFixed(2)}
                    </span>
                    <span className="text-[var(--text-muted)]">{mol.n_contacts}c</span>
                  </div>
                </button>
              );
            })}

            {filteredMolecules.length === 0 && (
              <div className="lab-card rounded-xl p-4 text-center text-xs text-[var(--text-muted)]">
                No curated targets match this filter.
              </div>
            )}
          </div>
        </aside>

        {/* CENTER STAGE: Full Interactive 3D WebGL Complex Viewport */}
        <section className="lg:col-span-5 xl:col-span-6 flex flex-col gap-3">
          {complex && (
            <div className="lab-card rounded-2xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="font-bold text-sm sm:text-base text-[var(--text-primary)] truncate">
                    {complex.title}
                  </h2>
                  <span className="font-mono-num text-[11px] font-bold px-2 py-0.5 rounded-md bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 border border-cyan-500/25">
                    PDB {complex.pdb_id}
                  </span>
                  <span className="text-[11px] font-medium px-2 py-0.5 rounded-md bg-pink-500/12 text-pink-600 dark:text-pink-300 border border-pink-500/25">
                    {complex.disease_tag}
                  </span>
                </div>
                <p className="text-[11px] text-[var(--text-muted)] mt-0.5 truncate">
                  {complex.family} · Clinical precedent: {complex.clinical_precedent}
                </p>
              </div>

              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  onClick={handleDownloadSdf}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold lab-card hover:border-cyan-500/50 text-[var(--text-primary)] cursor-pointer"
                >
                  <Download className="h-3.5 w-3.5 text-emerald-500" /> 3D SDF
                </button>
                <button
                  type="button"
                  onClick={handleCopySmiles}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] font-semibold lab-card hover:border-cyan-500/50 text-[var(--text-primary)] cursor-pointer"
                >
                  {copiedSmiles ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-500" /> Copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5 text-cyan-500" /> SMILES
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          <Mol3DViewer
            ref={viewerRef}
            complex={complex}
            appTheme={theme}
            className="h-[520px] lg:h-[calc(100vh-11.5rem)] min-h-[480px] w-full shadow-2xl"
          />
        </section>

        {/* RIGHT INSPECTOR: 2D SVG Formula, CNS Safety Dashboard & Clickable 3D Contacts */}
        <aside className="lg:col-span-4 xl:col-span-3.5 space-y-3 lg:h-[calc(100vh-5.5rem)] lg:overflow-y-auto pr-0.5">
          {loading || !complex ? (
            <div className="lab-card rounded-2xl p-8 text-center text-xs text-[var(--text-muted)]">
              Loading 2D structure &amp; 3D binding telemetry…
            </div>
          ) : (
            <>
              {/* 2D RDKit Vector Formula Card */}
              <div className="lab-card rounded-2xl p-3.5 space-y-2.5">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
                      2D Chemical Structure ({complex.lead_id})
                    </span>
                    <div className="font-mono-num text-xs font-semibold text-cyan-600 dark:text-cyan-400">
                      {complex.descriptors.formula}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setZoom2D(true)}
                    title="Expand 2D Chemical Formula"
                    className="p-1.5 rounded-lg lab-card hover:border-cyan-500/50 text-[var(--text-secondary)] cursor-pointer"
                  >
                    <Maximize2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                <div
                  onClick={() => setZoom2D(true)}
                  className="rounded-xl bg-[var(--bg-elevated)]/70 border border-[var(--border-subtle)] p-2 cursor-zoom-in rdkit-svg-wrap"
                  dangerouslySetInnerHTML={{
                    __html: theme === 'dark' ? complex.svg_2d_dark : complex.svg_2d_light,
                  }}
                />

                <button
                  type="button"
                  onClick={handleCopySmiles}
                  title="Click to copy canonical SMILES"
                  className="w-full text-left px-2.5 py-1.5 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)] hover:border-cyan-500/50 font-mono-num text-[10.5px] text-[var(--text-secondary)] break-all flex items-center justify-between gap-2 cursor-pointer"
                >
                  <span className="truncate">{complex.canonical_smiles}</span>
                  {copiedSmiles ? (
                    <Check className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                  ) : (
                    <Copy className="h-3.5 w-3.5 text-[var(--text-muted)] shrink-0" />
                  )}
                </button>

                {/* Descriptor pills */}
                <div className="grid grid-cols-4 gap-1.5 text-center font-mono-num text-[10px]">
                  {[
                    ['MW', `${complex.descriptors.mw.toFixed(1)}`],
                    ['LogP', `${complex.descriptors.clogp.toFixed(2)}`],
                    ['TPSA', `${complex.descriptors.tpsa.toFixed(1)}`],
                    ['pKa', `${complex.descriptors.most_basic_pka.toFixed(1)}`],
                    ['HBD/A', `${complex.descriptors.hbd}/${complex.descriptors.hba}`],
                    ['RotB', `${complex.descriptors.rot_bonds}`],
                    ['QED', `${complex.descriptors.qed.toFixed(3)}`],
                    ['SA', `${complex.descriptors.sa_score.toFixed(2)}`],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      className="rounded-lg bg-[var(--bg-elevated)]/80 border border-[var(--border-subtle)] py-1 px-1"
                    >
                      <div className="text-[9px] text-[var(--text-muted)]">{k}</div>
                      <div className="font-bold text-[var(--text-primary)]">{v}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Physics & Medicinal Chemistry KPI Dashboard */}
              <div className="lab-card rounded-2xl p-3.5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
                    3D Physics &amp; CNS Safety Profile
                  </span>
                  {complex.physics_3d.posebusters_passed && (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30">
                      <ShieldCheck className="h-3 w-3" /> PoseBusters PASS
                    </span>
                  )}
                </div>

                {/* Primary 4 physics tiles */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-xl bg-[var(--bg-elevated)]/80 border border-[var(--border-subtle)] p-2.5">
                    <div className="text-[10px] text-[var(--text-muted)] font-semibold">Vina ΔG</div>
                    <div className="font-mono-num text-lg font-bold text-emerald-500 dark:text-emerald-400">
                      {complex.physics_3d.top1_vina_dg.toFixed(2)}
                      <span className="text-[10px] font-normal text-[var(--text-muted)]"> kcal/mol</span>
                    </div>
                    <div className="font-mono-num text-[10px] text-[var(--text-muted)]">
                      LE {complex.physics_3d.le.toFixed(3)} · LLE {complex.physics_3d.lle_vina.toFixed(2)}
                    </div>
                  </div>

                  <div className="rounded-xl bg-[var(--bg-elevated)]/80 border border-[var(--border-subtle)] p-2.5">
                    <div className="text-[10px] text-[var(--text-muted)] font-semibold">
                      {complex.physics_3d.ddg_vs_drd2_mean !== null
                        ? 'ΔΔG vs DRD2'
                        : 'Canonical Anchor'}
                    </div>
                    <div className="font-mono-num text-lg font-bold text-cyan-500 dark:text-cyan-400">
                      {complex.physics_3d.ddg_vs_drd2_mean !== null
                        ? `${complex.physics_3d.ddg_vs_drd2_mean > 0 ? '+' : ''}${complex.physics_3d.ddg_vs_drd2_mean.toFixed(2)}`
                        : complex.physics_3d.anchor_min_dist !== null
                          ? `${complex.physics_3d.anchor_min_dist.toFixed(2)} Å`
                          : '—'}
                    </div>
                    <div className="font-mono-num text-[10px] text-[var(--text-muted)] truncate">
                      {complex.physics_3d.anchor_label}
                      {complex.physics_3d.anchor_min_dist !== null &&
                      complex.physics_3d.ddg_vs_drd2_mean !== null
                        ? ` (${complex.physics_3d.anchor_min_dist.toFixed(2)} Å)`
                        : ''}
                    </div>
                  </div>
                </div>

                {/* Calibrated ADMET & AD Risk Meters */}
                <div className="space-y-2">
                  {[
                    {
                      label: 'P(BBB+) Penetration',
                      val: complex.admet.bbb_prob,
                      goodHigh: true,
                      text: `${(complex.admet.bbb_prob * 100).toFixed(1)}%`,
                    },
                    {
                      label: 'P(Pgp−) Low Efflux',
                      val: complex.admet.pgp_minus_prob,
                      goodHigh: true,
                      text: `${(complex.admet.pgp_minus_prob * 100).toFixed(1)}%`,
                    },
                    {
                      label: 'P(hERG) Cardiotox Risk',
                      val: complex.admet.herg_prob,
                      goodHigh: false,
                      text: `${(complex.admet.herg_prob * 100).toFixed(1)}%`,
                    },
                    {
                      label: 'Min AD Similarity (≥0.30)',
                      val: Math.min(1, complex.admet.min_ad_sim),
                      goodHigh: true,
                      text: complex.admet.min_ad_sim.toFixed(2),
                    },
                  ].map((m) => {
                    const isGood = m.goodHigh ? m.val >= 0.3 : m.val <= 0.3;
                    const barColor = m.goodHigh
                      ? m.val >= 0.7 || m.label.includes('AD')
                        ? 'bg-emerald-500'
                        : 'bg-amber-500'
                      : m.val <= 0.25
                        ? 'bg-emerald-500'
                        : m.val <= 0.45
                          ? 'bg-amber-500'
                          : 'bg-rose-500';
                    return (
                      <div key={m.label} className="space-y-0.5">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-[var(--text-secondary)]">{m.label}</span>
                          <span
                            className={`font-mono-num font-bold ${
                              isGood ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-500'
                            }`}
                          >
                            {m.text}
                          </span>
                        </div>
                        <div className="h-1.5 rounded-full bg-slate-500/20 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${barColor} transition-all`}
                            style={{ width: `${Math.max(4, Math.min(100, m.val * 100))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Pfizer CNS MPO v2 + 6 Component Breakdown */}
                <div className="pt-2 border-t border-[var(--border-subtle)] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold text-[var(--text-primary)]">
                      Pfizer CNS MPO v2
                    </span>
                    <span className="font-mono-num text-xs font-extrabold text-cyan-600 dark:text-cyan-400">
                      {complex.admet.cns_mpo_v2.toFixed(2)} / 6.00
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                    {complex.cns_mpo_components
                      .filter((c) => c.weight > 0)
                      .map((c) => (
                        <div key={c.key} className="space-y-0.5">
                          <div className="flex justify-between font-mono-num text-[10px]">
                            <span className="text-[var(--text-muted)]">{c.label}</span>
                            <span className="text-[var(--text-primary)]">
                              {c.value}
                              {c.unit} ({c.t0.toFixed(2)})
                            </span>
                          </div>
                          <div className="h-1 rounded-full bg-slate-500/20 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400"
                              style={{ width: `${Math.max(4, c.t0 * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Strict Novelty & Nearest ChEMBL */}
                <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between text-[11px]">
                  <span className="text-[var(--text-muted)]">Max ChEMBL Tanimoto</span>
                  <div className="flex items-center gap-2 font-mono-num">
                    <span className="font-bold text-emerald-600 dark:text-emerald-400">
                      {complex.novelty.max_tanimoto_all_chembl.toFixed(3)}
                    </span>
                    {complex.novelty.nearest_all_chembl && (
                      <a
                        href={`https://www.ebi.ac.uk/chembl/compound_report_card/${complex.novelty.nearest_all_chembl.chembl_id}/`}
                        target="_blank"
                        rel="noreferrer"
                        className="text-cyan-600 dark:text-cyan-400 hover:underline inline-flex items-center gap-0.5"
                      >
                        {complex.novelty.nearest_all_chembl.chembl_id}
                        <ExternalLink className="h-2.5 w-2.5" />
                      </a>
                    )}
                  </div>
                </div>
              </div>

              {/* Interactive 3D Non-Covalent Contacts Table (Click to Zoom!) */}
              <div className="lab-card rounded-2xl p-3.5 space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10.5px] font-bold uppercase tracking-wider text-[var(--text-muted)]">
                    3D Non-Covalent Contacts ({complex.profile.interactions.length})
                  </span>
                  <span className="text-[10px] text-cyan-600 dark:text-cyan-400 flex items-center gap-1">
                    <Crosshair className="h-3 w-3" /> Click row to zoom 3D
                  </span>
                </div>

                <div className="max-h-56 overflow-y-auto rounded-xl border border-[var(--border-subtle)]">
                  <table className="w-full text-left text-[11px]">
                    <thead className="sticky top-0 bg-[var(--bg-elevated)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                      <tr>
                        <th className="py-1.5 px-2">Type</th>
                        <th className="py-1.5 px-2">Residue (Atom)</th>
                        <th className="py-1.5 px-2 text-right">Dist</th>
                        <th className="py-1.5 px-2 text-right">Angle</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--border-subtle)] font-mono-num">
                      {complex.profile.interactions.map((rec, idx) => {
                        const meta = INTERACTION_META[rec.kind] ?? INTERACTION_META.hbond;
                        const col = theme === 'dark' ? meta.dark : meta.light;
                        const selected = activeContactIdx === idx;
                        return (
                          <tr
                            key={`${rec.kind}-${rec.protein_residue}-${rec.protein_atom}-${idx}`}
                            onClick={() => {
                              setActiveContactIdx(idx);
                              viewerRef.current?.focusContact(rec);
                            }}
                            className={`cursor-pointer transition-colors ${
                              selected
                                ? 'bg-cyan-500/18'
                                : 'hover:bg-slate-500/10'
                            }`}
                          >
                            <td className="py-1.5 px-2">
                              <span
                                className="inline-block px-1.5 py-0.5 rounded text-[9.5px] font-bold"
                                style={{ backgroundColor: `${col}22`, color: col }}
                              >
                                {meta.short}
                              </span>
                            </td>
                            <td className="py-1.5 px-2 text-[var(--text-primary)]">
                              {rec.protein_residue}{' '}
                              <span className="text-[var(--text-muted)]">({rec.protein_atom})</span>
                            </td>
                            <td className="py-1.5 px-2 text-right font-semibold text-[var(--text-primary)]">
                              {rec.distance_angstrom.toFixed(2)} Å
                            </td>
                            <td className="py-1.5 px-2 text-right text-[var(--text-muted)]">
                              {rec.angle_deg !== null && rec.angle_deg !== undefined
                                ? `${rec.angle_deg.toFixed(0)}°`
                                : '—'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {complex.triage_note && (
                  <div className="rounded-xl bg-sky-500/10 border border-sky-500/25 p-2.5 text-[11px] text-sky-700 dark:text-sky-300 flex items-start gap-2">
                    <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                    <span>{complex.triage_note}</span>
                  </div>
                )}

                <p className="text-[10.5px] text-[var(--text-muted)] leading-relaxed">
                  {complex.notes}
                </p>
              </div>
            </>
          )}
        </aside>
      </div>

      {/* 2D Chemical Formula Zoom Modal */}
      {zoom2D && complex && (
        <div
          onClick={() => setZoom2D(false)}
          className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="lab-card rounded-2xl max-w-2xl w-full p-5 space-y-4 shadow-2xl"
          >
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="font-bold text-base text-[var(--text-primary)]">
                  {complex.lead_id} · {complex.title} ({complex.pdb_id})
                </h3>
                <p className="font-mono-num text-xs text-cyan-600 dark:text-cyan-400">
                  {complex.descriptors.formula} · MW {complex.descriptors.mw.toFixed(2)} · LogP{' '}
                  {complex.descriptors.clogp.toFixed(2)} · TPSA {complex.descriptors.tpsa.toFixed(1)} Å²
                </p>
              </div>
              <button
                type="button"
                onClick={() => setZoom2D(false)}
                className="p-1.5 rounded-lg lab-card hover:border-rose-500/50 text-[var(--text-secondary)] cursor-pointer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div
              className="rounded-xl bg-[var(--bg-elevated)] border border-[var(--border-subtle)] p-4 rdkit-svg-wrap"
              dangerouslySetInnerHTML={{
                __html: theme === 'dark' ? complex.svg_2d_dark : complex.svg_2d_light,
              }}
            />

            <div className="space-y-1.5 text-xs">
              <div className="font-semibold text-[var(--text-muted)]">Canonical SMILES</div>
              <div className="p-2.5 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)] font-mono-num text-[11px] break-all text-[var(--text-primary)]">
                {complex.canonical_smiles}
              </div>
              <div className="font-semibold text-[var(--text-muted)] pt-1">
                Physiological pH 7.4 Protonation State
              </div>
              <div className="p-2.5 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-subtle)] font-mono-num text-[11px] break-all text-[var(--text-secondary)]">
                {complex.smiles_ph74}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
