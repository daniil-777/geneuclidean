import {
  Camera,
  Crosshair,
  Eye,
  Focus,
  Layers,
  Maximize2,
  Moon,
  RotateCcw,
  RotateCw,
  Sun,
} from 'lucide-react';
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import type { ComplexDetail, InteractionKind, InteractionRecord, ThemeMode } from '../types/studio';

export const INTERACTION_META: Record<
  InteractionKind,
  { label: string; short: string; dark: string; light: string }
> = {
  salt_bridge: { label: 'Salt Bridge', short: 'SB', dark: '#F472B6', light: '#DB2777' },
  hbond: { label: 'H-Bond', short: 'HB', dark: '#38BDF8', light: '#0284C7' },
  pi_stacking: { label: 'π–π Stack', short: 'π–π', dark: '#FBBF24', light: '#D97706' },
  cation_pi: { label: 'Cation–π', short: 'CAT-π', dark: '#FB923C', light: '#EA580C' },
  hydrophobic: { label: 'Hydrophobic', short: 'HYD', dark: '#34D399', light: '#059669' },
  halogen_bond: { label: 'Halogen', short: 'XB', dark: '#A78BFA', light: '#7C3AED' },
};

const ALL_KINDS: InteractionKind[] = [
  'salt_bridge',
  'hbond',
  'pi_stacking',
  'cation_pi',
  'hydrophobic',
  'halogen_bond',
];

const HYDROPATHY: Record<string, number> = {
  ILE: 4.5, VAL: 4.2, LEU: 3.8, PHE: 2.8, CYS: 2.5, MET: 1.9, ALA: 1.8,
  GLY: -0.4, THR: -0.7, SER: -0.8, TRP: -0.9, TYR: -1.3, PRO: -1.6,
  HIS: -3.2, GLU: -3.5, GLN: -3.5, ASP: -3.5, ASN: -3.5, LYS: -3.9, ARG: -4.5,
};

function lerpHex(c1: [number, number, number], c2: [number, number, number], t: number): string {
  const k = Math.max(0, Math.min(1, t));
  const r = Math.round(c1[0] + (c2[0] - c1[0]) * k);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * k);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * k);
  return `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

function attentionHex(score: number, canvasTheme: ThemeMode): string {
  const s = Math.max(0, Math.min(1, score || 0));
  const stops: [number, [number, number, number]][] =
    canvasTheme === 'dark'
      ? [
          [0.0, [30, 27, 75]],
          [0.36, [79, 70, 229]],
          [0.72, [245, 158, 11]],
          [1.0, [253, 224, 71]],
        ]
      : [
          [0.0, [203, 213, 225]],
          [0.36, [99, 102, 241]],
          [0.72, [234, 88, 12]],
          [1.0, [180, 83, 9]],
        ];
  for (let i = 0; i < stops.length - 1; i++) {
    if (s <= stops[i + 1][0]) {
      const t = (s - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return lerpHex(stops[i][1], stops[i + 1][1], t);
    }
  }
  return canvasTheme === 'dark' ? '#FDE047' : '#B45309';
}

function hydroHex(resn: string): string {
  const h = HYDROPATHY[(resn || '').toUpperCase()] ?? 0;
  const t = (h + 4.5) / 9.0;
  return t < 0.5
    ? lerpHex([56, 189, 248], [148, 163, 184], t * 2)
    : lerpHex([148, 163, 184], [245, 158, 11], (t - 0.5) * 2);
}

function lookupAttn(
  atom: { resn?: string; resi?: number | string; chain?: string },
  attnMap: Record<string, number>,
): number {
  const resn = (atom.resn || '').toUpperCase();
  const resi = String(atom.resi ?? '');
  const chain = (atom.chain || 'A').trim() || 'A';
  const full = `${resn}${resi}:${chain}`;
  const shortKey = `${resn}${resi}`;
  if (full in attnMap) return attnMap[full];
  if (shortKey in attnMap) return attnMap[shortKey];
  if (resi in attnMap) return attnMap[resi];
  return 0.12;
}

function parseResidueSel(resStr: string): { resi?: number; chain?: string } {
  const [left, chain] = resStr.split(':');
  const match = left.match(/(\d+)/);
  const sel: { resi?: number; chain?: string } = {};
  if (match) sel.resi = Number.parseInt(match[1], 10);
  if (chain) sel.chain = chain;
  return sel;
}

export interface Mol3DViewerHandle {
  focusContact: (rec: InteractionRecord) => void;
  focusAnchor: () => void;
  resetView: () => void;
  exportPng: () => void;
}

interface Mol3DViewerProps {
  complex: ComplexDetail | null;
  appTheme: ThemeMode;
  compact?: boolean;
  initialSpin?: boolean;
  className?: string;
}

type LigandStyle = 'ballstick' | 'stick' | 'sphere';
type ColorMode = 'attention' | 'hydrophobicity' | 'chain';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Viewer3D = any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Lib3Dmol = any;

export const Mol3DViewer = forwardRef<Mol3DViewerHandle, Mol3DViewerProps>(function Mol3DViewer(
  { complex, appTheme, compact = false, initialSpin = false, className = '' },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const libRef = useRef<Lib3Dmol | null>(null);
  const viewerRef = useRef<Viewer3D | null>(null);
  const modelsRef = useRef<{
    protein: Viewer3D | null;
    ligand: Viewer3D | null;
    native: Viewer3D | null;
    surface: Viewer3D | null;
  }>({ protein: null, ligand: null, native: null, surface: null });

  const [libReady, setLibReady] = useState(false);
  const [webglError, setWebglError] = useState<string | null>(null);

  const [canvasTheme, setCanvasTheme] = useState<ThemeMode>(appTheme);
  const [ligandStyle, setLigandStyle] = useState<LigandStyle>('ballstick');
  const [colorMode, setColorMode] = useState<ColorMode>('attention');
  const [showSurface, setShowSurface] = useState(true);
  const [showSidechains, setShowSidechains] = useState(true);
  const [showNative, setShowNative] = useState(!compact);
  const [showLabels, setShowLabels] = useState(true);
  const [spinning, setSpinning] = useState(initialSpin);
  const [visibleKinds, setVisibleKinds] = useState<Record<InteractionKind, boolean>>({
    salt_bridge: true,
    hbond: true,
    pi_stacking: true,
    cation_pi: true,
    hydrophobic: !compact,
    halogen_bond: true,
  });

  // Sync canvas theme when global app theme switches
  useEffect(() => {
    setCanvasTheme(appTheme);
  }, [appTheme]);

  // Dynamically import 3dmol once (code-split chunk, zero external CDN flakiness)
  useEffect(() => {
    let alive = true;
    import('3dmol')
      .then((mod) => {
        if (!alive) return;
        // 3dmol exports either default or named namespace depending on bundler interop
        libRef.current = (mod as { default?: Lib3Dmol }).default ?? mod;
        setLibReady(true);
      })
      .catch((err) => {
        if (!alive) return;
        setWebglError(err instanceof Error ? err.message : 'Failed to initialize 3Dmol WebGL engine.');
      });
    return () => {
      alive = false;
    };
  }, []);

  const applyStyling = useCallback(() => {
    const viewer = viewerRef.current;
    const $3Dmol = libRef.current;
    const { protein, ligand, native } = modelsRef.current;
    if (!viewer || !$3Dmol || !complex) return;

    const bgHex = canvasTheme === 'dark' ? '#050814' : '#F8FAFC';
    viewer.setBackgroundColor(bgHex, 1.0);

    const hasAttn = complex.attention && Object.keys(complex.attention).length > 0;
    const colorfn = (atom: { resn?: string; resi?: number | string; chain?: string }) => {
      if (colorMode === 'attention' && hasAttn) {
        return attentionHex(lookupAttn(atom, complex.attention), canvasTheme);
      }
      if (colorMode === 'hydrophobicity') {
        return hydroHex(atom.resn || '');
      }
      const chainMap: Record<string, string> =
        canvasTheme === 'dark'
          ? { A: '#64748B', R: '#64748B', B: '#6366F1', C: '#0EA5E9', D: '#14B8A6' }
          : { A: '#94A3B8', R: '#94A3B8', B: '#4F46E5', C: '#0284C7', D: '#0D9488' };
      return chainMap[(atom.chain || 'A').trim()] || '#64748B';
    };

    if (protein) {
      protein.setStyle({}, { cartoon: { colorfunc: colorfn, opacity: 0.78, thickness: 0.36 } });
      if (showSidechains) {
        const keyRes = complex.profile?.key_residues ?? [];
        keyRes.slice(0, 14).forEach((resStr) => {
          protein.addStyle(parseResidueSel(resStr), {
            stick: { radius: 0.11, colorscheme: 'Jmol', opacity: 0.88 },
          });
        });
      }
    }

    if (modelsRef.current.surface !== null) {
      try {
        viewer.removeSurface(modelsRef.current.surface);
      } catch {
        // ignore surface removal errors on re-render
      }
      modelsRef.current.surface = null;
    }
    if (showSurface && protein) {
      modelsRef.current.surface = viewer.addSurface(
        $3Dmol.SurfaceType.SAS,
        { opacity: canvasTheme === 'dark' ? 0.22 : 0.16, colorfunc: colorfn },
        { model: protein },
      );
    }

    const cpkMap =
      canvasTheme === 'dark'
        ? {
            C: '#10B981', N: '#3B82F6', O: '#EF4444', S: '#FACC15',
            F: '#4ADE80', Cl: '#22C55E', Br: '#A855F7', I: '#C084FC', P: '#F97316', H: '#E2E8F0',
          }
        : {
            C: '#0F766E', N: '#2563EB', O: '#DC2626', S: '#CA8A04',
            F: '#16A34A', Cl: '#15803D', Br: '#7E22CE', I: '#6B21A8', P: '#EA580C', H: '#CBD5E1',
          };

    if (ligand) {
      if (ligandStyle === 'ballstick') {
        ligand.setStyle(
          {},
          {
            stick: { radius: 0.17, colorscheme: { prop: 'elem', map: cpkMap } },
            sphere: { scale: 0.26, colorscheme: { prop: 'elem', map: cpkMap } },
          },
        );
      } else if (ligandStyle === 'stick') {
        ligand.setStyle({}, { stick: { radius: 0.2, colorscheme: { prop: 'elem', map: cpkMap } } });
      } else {
        ligand.setStyle({}, { sphere: { scale: 0.72, colorscheme: { prop: 'elem', map: cpkMap } } });
      }
    }

    if (native) {
      if (showNative) {
        native.setStyle(
          {},
          {
            stick: {
              radius: 0.12,
              color: canvasTheme === 'dark' ? '#38BDF8' : '#0284C7',
              opacity: 0.42,
            },
          },
        );
      } else {
        native.setStyle({}, {});
      }
    }

    // Redraw 6-channel 3D dashed interaction cylinders + Ångström labels
    viewer.removeAllShapes();
    viewer.removeAllLabels();
    const interactions = complex.profile?.interactions ?? [];
    const labeledRes = new Set<string>();
    const labelBg = canvasTheme === 'dark' ? 'rgba(6, 9, 19, 0.84)' : 'rgba(255, 255, 255, 0.92)';
    const resFont = canvasTheme === 'dark' ? '#F8FAFC' : '#0F172A';

    interactions.forEach((rec) => {
      if (!visibleKinds[rec.kind]) return;
      const meta = INTERACTION_META[rec.kind] ?? INTERACTION_META.hbond;
      const col = canvasTheme === 'dark' ? meta.dark : meta.light;
      const p = { x: rec.protein_xyz[0], y: rec.protein_xyz[1], z: rec.protein_xyz[2] };
      const l = { x: rec.ligand_xyz[0], y: rec.ligand_xyz[1], z: rec.ligand_xyz[2] };

      viewer.addCylinder({
        start: p,
        end: l,
        radius: rec.kind === 'hydrophobic' ? 0.05 : 0.075,
        dashed: true,
        dashLength: 0.24,
        gapLength: 0.16,
        color: col,
        opacity: rec.kind === 'hydrophobic' ? 0.62 : 0.96,
      });

      if (showLabels && rec.kind !== 'hydrophobic') {
        viewer.addLabel(`${rec.distance_angstrom.toFixed(2)} Å`, {
          position: { x: 0.5 * (p.x + l.x), y: 0.5 * (p.y + l.y), z: 0.5 * (p.z + l.z) },
          fontSize: compact ? 9 : 10,
          fontColor: col,
          backgroundColor: labelBg,
          borderColor: col,
          borderThickness: 1,
          backgroundOpacity: 0.88,
          inFront: true,
        });
      }

      if (showSidechains && !labeledRes.has(rec.protein_residue) && rec.kind !== 'hydrophobic') {
        labeledRes.add(rec.protein_residue);
        viewer.addLabel(rec.protein_residue.replace(/:[A-Z]$/, ''), {
          position: p,
          fontSize: compact ? 10 : 11,
          fontColor: resFont,
          backgroundColor: labelBg,
          borderColor: col,
          borderThickness: 1,
          backgroundOpacity: 0.9,
          inFront: true,
        });
      }
    });

    viewer.render();
  }, [
    complex,
    canvasTheme,
    colorMode,
    showSurface,
    showSidechains,
    ligandStyle,
    showNative,
    showLabels,
    visibleKinds,
    compact,
  ]);

  // Mount viewer & load PDB/SDF when complex changes
  useEffect(() => {
    if (!libReady || !containerRef.current || !complex) return;
    const $3Dmol = libRef.current;
    if (!$3Dmol) return;

    try {
      if (!viewerRef.current) {
        viewerRef.current = $3Dmol.createViewer(containerRef.current, {
          backgroundColor: canvasTheme === 'dark' ? '#050814' : '#F8FAFC',
          antialias: true,
          cartoonQuality: 8,
        });
      }
      const viewer = viewerRef.current;
      viewer.clear();
      modelsRef.current = { protein: null, ligand: null, native: null, surface: null };

      if (complex.pdb) {
        modelsRef.current.protein = viewer.addModel(complex.pdb, 'pdb');
      }
      if (complex.ligand_sdf) {
        modelsRef.current.ligand = viewer.addModel(complex.ligand_sdf, 'sdf');
      }
      if (complex.native_sdf) {
        modelsRef.current.native = viewer.addModel(complex.native_sdf, 'sdf');
      }

      applyStyling();

      if (modelsRef.current.ligand) {
        viewer.zoomTo({ model: modelsRef.current.ligand }, 0);
        viewer.zoom(compact ? 0.92 : 0.84, 0);
      } else {
        viewer.zoomTo({}, 0);
      }
      viewer.render();
      viewer.spin(spinning ? 'y' : false, compact ? 0.35 : 0.3);
    } catch (err) {
      setWebglError(err instanceof Error ? err.message : 'WebGL initialization failed.');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [libReady, complex?.target, complex?.lead_id]);

  // Re-apply visual styles without rebuilding models
  useEffect(() => {
    if (viewerRef.current && complex) {
      applyStyling();
    }
  }, [applyStyling, complex]);

  // Spin toggle effect
  useEffect(() => {
    if (viewerRef.current) {
      viewerRef.current.spin(spinning ? 'y' : false, compact ? 0.35 : 0.3);
    }
  }, [spinning, compact]);

  // Resize observer so WebGL canvas stays crisp across layout/viewport changes
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => {
      if (viewerRef.current) {
        viewerRef.current.resize();
        viewerRef.current.render();
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Imperative camera API
  const focusContact = useCallback((rec: InteractionRecord) => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    setSpinning(false);
    viewer.spin(false);
    const mx = 0.5 * (rec.protein_xyz[0] + rec.ligand_xyz[0]);
    const my = 0.5 * (rec.protein_xyz[1] + rec.ligand_xyz[1]);
    const mz = 0.5 * (rec.protein_xyz[2] + rec.ligand_xyz[2]);
    viewer.zoomTo({ model: 0, ...parseResidueSel(rec.protein_residue) }, 420);
    viewer.center({ x: mx, y: my, z: mz }, 380);
  }, []);

  const focusAnchor = useCallback(() => {
    if (!complex) return;
    const anchorRec =
      complex.profile.interactions.find((r) => r.kind === 'salt_bridge') ??
      complex.profile.interactions.find((r) => r.kind === 'hbond') ??
      complex.profile.interactions[0];
    if (anchorRec) {
      focusContact(anchorRec);
    } else if (viewerRef.current && modelsRef.current.ligand) {
      viewerRef.current.zoomTo({ model: modelsRef.current.ligand }, 380);
    }
  }, [complex, focusContact]);

  const resetView = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    setSpinning(false);
    viewer.spin(false);
    if (modelsRef.current.ligand) {
      viewer.zoomTo({ model: modelsRef.current.ligand }, 360);
      viewer.zoom(0.84, 360);
    } else {
      viewer.zoomTo({}, 360);
    }
  }, []);

  const exportPng = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer || !complex) return;
    const uri = viewer.pngURI();
    const a = document.createElement('a');
    a.href = uri;
    a.download = `GenEuclidean_${complex.lead_id}_${complex.pdb_id}_3d.png`;
    a.click();
  }, [complex]);

  useImperativeHandle(
    ref,
    () => ({ focusContact, focusAnchor, resetView, exportPng }),
    [focusContact, focusAnchor, resetView, exportPng],
  );

  const counts = complex?.profile?.counts_by_type ?? {};
  const hasNative = Boolean(complex?.native_sdf);

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-[var(--border-subtle)] ${
        canvasTheme === 'dark' ? 'bg-[#050814]' : 'bg-slate-50'
      } ${className}`}
    >
      {/* 3Dmol WebGL stage */}
      <div ref={containerRef} className="absolute inset-0 w-full h-full" />

      {/* Loading / Error overlay */}
      {(!libReady || !complex) && !webglError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-slate-400 bg-slate-950/50 backdrop-blur-sm z-10">
          <div className="h-6 w-6 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
          <span>Initializing 3D WebGL Protein–Ligand Complex…</span>
        </div>
      )}
      {webglError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-xs text-rose-300 bg-slate-950/85 z-20">
          <p className="font-semibold mb-1">WebGL 3D Viewport Unavailable</p>
          <p className="text-slate-400 max-w-sm">{webglError}</p>
        </div>
      )}

      {/* Full studio floating toolbar */}
      {!compact && complex && (
        <>
          <div className="absolute top-3 left-3 right-3 z-20 flex flex-wrap items-start justify-between gap-2 pointer-events-none">
            {/* Representation controls */}
            <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-950/80 dark:bg-slate-950/82 backdrop-blur-md border border-slate-700/60 text-slate-100 shadow-lg">
              <div className="inline-flex rounded-lg bg-slate-900/90 p-0.5 border border-slate-700/60">
                {(
                  [
                    ['ballstick', 'Ball+Stick'],
                    ['stick', 'Stick'],
                    ['sphere', 'Sphere'],
                  ] as const
                ).map(([val, lbl]) => (
                  <button
                    key={val}
                    type="button"
                    onClick={() => setLigandStyle(val)}
                    className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors cursor-pointer ${
                      ligandStyle === val
                        ? 'bg-cyan-500/25 text-cyan-200'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {lbl}
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={() => setShowSurface((v) => !v)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors flex items-center gap-1 cursor-pointer ${
                  showSurface
                    ? 'bg-cyan-500/20 border-cyan-400/50 text-cyan-200'
                    : 'bg-slate-900/70 border-slate-700/60 text-slate-300 hover:border-slate-500'
                }`}
              >
                <Layers className="h-3 w-3" /> Surface
              </button>

              <button
                type="button"
                onClick={() => setShowSidechains((v) => !v)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors cursor-pointer ${
                  showSidechains
                    ? 'bg-cyan-500/20 border-cyan-400/50 text-cyan-200'
                    : 'bg-slate-900/70 border-slate-700/60 text-slate-300 hover:border-slate-500'
                }`}
              >
                Residues
              </button>

              {hasNative && (
                <button
                  type="button"
                  onClick={() => setShowNative((v) => !v)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors flex items-center gap-1 cursor-pointer ${
                    showNative
                      ? 'bg-sky-500/20 border-sky-400/50 text-sky-200'
                      : 'bg-slate-900/70 border-slate-700/60 text-slate-300 hover:border-slate-500'
                  }`}
                >
                  <Eye className="h-3 w-3" /> Native Ghost
                </button>
              )}

              <select
                value={colorMode}
                onChange={(e) => setColorMode(e.target.value as ColorMode)}
                aria-label="Protein coloring mode"
                className="px-2 py-1 rounded-lg text-[11px] font-medium bg-slate-900 border border-slate-700/70 text-slate-100 outline-none focus:border-cyan-400 cursor-pointer"
              >
                <option value="attention">Color: 3D Perceiver Attention</option>
                <option value="hydrophobicity">Color: Hydrophobicity</option>
                <option value="chain">Color: Chain</option>
              </select>
            </div>

            {/* Camera & Export controls */}
            <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-950/80 backdrop-blur-md border border-slate-700/60 text-slate-100 shadow-lg">
              <button
                type="button"
                onClick={focusAnchor}
                title="Focus Binding Anchor"
                className="px-2 py-1 rounded-lg text-[11px] font-medium bg-slate-900/70 border border-slate-700/60 hover:border-cyan-400/60 text-slate-200 flex items-center gap-1 cursor-pointer"
              >
                <Crosshair className="h-3 w-3 text-pink-400" /> Anchor
              </button>
              <button
                type="button"
                onClick={() => {
                  if (viewerRef.current && modelsRef.current.ligand) {
                    viewerRef.current.zoomTo({ model: modelsRef.current.ligand }, 360);
                    viewerRef.current.zoom(0.86, 360);
                  }
                }}
                title="Zoom Binding Pocket"
                className="px-2 py-1 rounded-lg text-[11px] font-medium bg-slate-900/70 border border-slate-700/60 hover:border-cyan-400/60 text-slate-200 flex items-center gap-1 cursor-pointer"
              >
                <Focus className="h-3 w-3 text-cyan-400" /> Pocket
              </button>
              <button
                type="button"
                onClick={() => viewerRef.current?.zoomTo({}, 380)}
                title="Full Pocket View"
                className="px-2 py-1 rounded-lg text-[11px] font-medium bg-slate-900/70 border border-slate-700/60 hover:border-cyan-400/60 text-slate-200 flex items-center gap-1 cursor-pointer"
              >
                <Maximize2 className="h-3 w-3" /> All
              </button>
              <button
                type="button"
                onClick={resetView}
                title="Reset Camera"
                className="p-1.5 rounded-lg bg-slate-900/70 border border-slate-700/60 hover:border-cyan-400/60 text-slate-200 cursor-pointer"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setShowLabels((v) => !v)}
                className={`px-2 py-1 rounded-lg text-[11px] font-mono-num font-medium border cursor-pointer ${
                  showLabels
                    ? 'bg-cyan-500/20 border-cyan-400/50 text-cyan-200'
                    : 'bg-slate-900/70 border-slate-700/60 text-slate-400'
                }`}
              >
                Å
              </button>
              <button
                type="button"
                onClick={() => setSpinning((v) => !v)}
                className={`px-2 py-1 rounded-lg text-[11px] font-medium border flex items-center gap-1 cursor-pointer ${
                  spinning
                    ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                    : 'bg-slate-900/70 border-slate-700/60 text-slate-200 hover:border-cyan-400/60'
                }`}
              >
                <RotateCw className={`h-3 w-3 ${spinning ? 'animate-spin' : ''}`} /> Spin
              </button>
              <button
                type="button"
                onClick={() => setCanvasTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
                title="Toggle 3D Canvas Backdrop (Dark/Light)"
                className="p-1.5 rounded-lg bg-slate-900/70 border border-slate-700/60 hover:border-cyan-400/60 text-slate-200 cursor-pointer"
              >
                {canvasTheme === 'dark' ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              </button>
              <button
                type="button"
                onClick={exportPng}
                title="Download 3D Snapshot PNG"
                className="px-2 py-1 rounded-lg text-[11px] font-medium bg-cyan-500/20 border border-cyan-400/40 hover:bg-cyan-500/30 text-cyan-200 flex items-center gap-1 cursor-pointer"
              >
                <Camera className="h-3 w-3" /> PNG
              </button>
            </div>
          </div>

          {/* Bottom per-channel interaction toggles + Perceiver attention scale */}
          <div className="absolute bottom-3 left-3 right-3 z-20 flex flex-wrap items-end justify-between gap-2 pointer-events-none">
            <div className="pointer-events-auto flex flex-wrap items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-slate-950/80 backdrop-blur-md border border-slate-700/60 shadow-lg">
              {ALL_KINDS.map((kind) => {
                const meta = INTERACTION_META[kind];
                const cnt = counts[kind] ?? 0;
                const on = visibleKinds[kind];
                return (
                  <button
                    key={kind}
                    type="button"
                    onClick={() =>
                      setVisibleKinds((prev) => ({ ...prev, [kind]: !prev[kind] }))
                    }
                    className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border transition-all cursor-pointer ${
                      on
                        ? 'bg-slate-900/90 border-slate-600/80 text-slate-100'
                        : 'bg-slate-900/40 border-slate-800 text-slate-500 opacity-50'
                    }`}
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: meta.dark }}
                    />
                    {meta.label}
                    <span className="font-mono-num text-[10px] text-slate-400">({cnt})</span>
                  </button>
                );
              })}
            </div>

            <div className="pointer-events-auto flex items-center gap-3 px-3 py-1.5 rounded-xl bg-slate-950/80 backdrop-blur-md border border-slate-700/60 text-[11px] text-slate-300 shadow-lg">
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                De Novo ({complex.lead_id})
              </span>
              {hasNative && showNative && (
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full bg-sky-400" />
                  Native Ghost
                </span>
              )}
              {colorMode === 'attention' && (
                <span className="hidden sm:flex items-center gap-1.5 border-l border-slate-700 pl-2.5">
                  <span className="text-[10px] uppercase tracking-wider text-amber-300 font-semibold">
                    3D Attn
                  </span>
                  <span className="h-2 w-14 rounded-full bg-gradient-to-r from-indigo-950 via-indigo-500 via-65% via-amber-500 to-yellow-300" />
                </span>
              )}
            </div>
          </div>
        </>
      )}

      {/* Compact hero overlay controls */}
      {compact && complex && (
        <div className="absolute bottom-3 left-3 right-3 z-20 flex items-center justify-between gap-2 pointer-events-none">
          <div className="pointer-events-auto px-2.5 py-1 rounded-lg bg-slate-950/78 backdrop-blur-md border border-slate-700/60 text-[11px] text-slate-200 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono-num font-semibold text-cyan-300">{complex.lead_id}</span>
            <span className="text-slate-400">in {complex.target} ({complex.pdb_id})</span>
            <span className="hidden sm:inline font-mono-num text-emerald-300">
              {complex.physics_3d.top1_vina_dg.toFixed(2)} kcal/mol
            </span>
          </div>
          <div className="pointer-events-auto flex items-center gap-1">
            <button
              type="button"
              onClick={() => setSpinning((v) => !v)}
              title="Toggle Auto-Spin"
              className={`px-2 py-1 rounded-lg text-[11px] font-medium border backdrop-blur-md cursor-pointer ${
                spinning
                  ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                  : 'bg-slate-950/75 border-slate-700/60 text-slate-300'
              }`}
            >
              {spinning ? 'Pause' : 'Spin'}
            </button>
            <button
              type="button"
              onClick={resetView}
              title="Reset 3D Camera"
              className="p-1.5 rounded-lg bg-slate-950/75 border border-slate-700/60 text-slate-300 hover:text-white backdrop-blur-md cursor-pointer"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
});
