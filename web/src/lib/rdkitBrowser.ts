import { assetUrl } from './data';
import type { ThemeMode } from '../types/studio';

export interface SmartsRule {
  name: string;
  smarts: string;
  pka: number;
}

export interface EnsembleMember {
  w1_shape: [number, number];
  w1_int8_b64: string;
  w1_scales: number[];
  b1: number[];
  w2: number[];
  b2: number;
}

export interface CalibratedEndpointModel {
  descriptor_mean: number[];
  descriptor_std: number[];
  ensemble_members: EnsembleMember[];
  platt: { coef: number; intercept: number };
  cv_metrics?: Record<string, number>;
}

export interface BrowserAdmetPack {
  schema_version?: string;
  endpoints: Record<'bbb' | 'pgp' | 'herg', CalibratedEndpointModel>;
  pka_smarts_rules: {
    basic: SmartsRule[];
    acidic: SmartsRule[];
  };
}

export interface BrowserQsarPack {
  schema_version?: string;
  endpoints: Record<string, CalibratedEndpointModel>;
}

export interface LiveMoleculeScore {
  valid: boolean;
  canonicalSmiles: string;
  svgDark: string;
  svgLight: string;
  mw: number;
  clogp: number;
  clogd74: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotBonds: number;
  mostBasicPka: number;
  cnsMpoV1: number;
  cnsMpoV2: number;
  bbbProb: number;
  pgpProb: number;
  hergProb: number;
  jointCnsSafe: boolean;
  targetProbs: Record<string, number>;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RDKitModule = any;

declare global {
  interface Window {
    initRDKitModule?: (opts?: { locateFile?: (file: string) => string }) => Promise<RDKitModule>;
  }
}

const RDKIT_CDNS = [
  'https://unpkg.com/@rdkit/rdkit@2025.3.4-1.0.0/dist/',
  'https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2025.3.4-1.0.0/dist/',
];

let rdkitPromise: Promise<RDKitModule> | null = null;

export function loadRDKit(): Promise<RDKitModule> {
  if (rdkitPromise) return rdkitPromise;
  rdkitPromise = new Promise<RDKitModule>((resolve, reject) => {
    const tryCdn = (idx: number) => {
      if (idx >= RDKIT_CDNS.length) {
        reject(new Error('Failed to load RDKit.js WASM from CDN.'));
        return;
      }
      const base = RDKIT_CDNS[idx];
      if (typeof window.initRDKitModule === 'function') {
        window
          .initRDKitModule({ locateFile: (f: string) => `${base}${f}` })
          .then(resolve)
          .catch(() => tryCdn(idx + 1));
        return;
      }
      const script = document.createElement('script');
      script.src = `${base}RDKit_minimal.js`;
      script.async = true;
      script.onload = () => {
        if (typeof window.initRDKitModule === 'function') {
          window
            .initRDKitModule({ locateFile: (f: string) => `${base}${f}` })
            .then(resolve)
            .catch(() => tryCdn(idx + 1));
        } else {
          tryCdn(idx + 1);
        }
      };
      script.onerror = () => tryCdn(idx + 1);
      document.head.appendChild(script);
    };
    tryCdn(0);
  });
  rdkitPromise.catch(() => {
    rdkitPromise = null;
  });
  return rdkitPromise;
}

function decodeBase64Int8(b64: string): Int8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int8Array(bytes.buffer);
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, x))));
}

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

interface DecodedEndpoint {
  mean: number[];
  std: number[];
  members: {
    rows: number;
    cols: number;
    w1: Int8Array;
    scales: number[];
    b1: number[];
    w2: number[];
    b2: number;
  }[];
  platt: { coef: number; intercept: number };
}

function decodeEndpoint(ep: CalibratedEndpointModel): DecodedEndpoint {
  return {
    mean: ep.descriptor_mean,
    std: ep.descriptor_std,
    members: ep.ensemble_members.map((m) => ({
      rows: m.w1_shape[0],
      cols: m.w1_shape[1],
      w1: decodeBase64Int8(m.w1_int8_b64),
      scales: m.w1_scales,
      b1: m.b1,
      w2: m.w2,
      b2: m.b2,
    })),
    platt: ep.platt,
  };
}

function runEndpoint(ep: DecodedEndpoint, features: Float32Array): number {
  const x = new Float32Array(features);
  for (let i = 0; i < 12; i++) {
    const s = ep.std[i] > 1e-8 ? ep.std[i] : 1.0;
    x[1024 + i] = (x[1024 + i] - ep.mean[i]) / s;
  }
  let pSum = 0;
  for (const m of ep.members) {
    let out = m.b2;
    for (let r = 0; r < m.rows; r++) {
      let acc = 0;
      const off = r * m.cols;
      for (let c = 0; c < m.cols; c++) acc += m.w1[off + c] * x[c];
      const h = silu(acc * m.scales[r] + m.b1[r]);
      out += m.w2[r] * h;
    }
    pSum += sigmoid(out);
  }
  const pRaw = Math.max(1e-5, Math.min(1 - 1e-5, pSum / Math.max(1, ep.members.length)));
  const logit = Math.log(pRaw / (1 - pRaw));
  return sigmoid(ep.platt.coef * logit + ep.platt.intercept);
}

// Exact Wager (2010/2016) Pfizer CNS MPO desirability functions
function t0ClogP(v: number): number {
  return v <= 3 ? 1 : v >= 5 ? 0 : (5 - v) / 2;
}
function t0ClogD(v: number): number {
  return v <= 2 ? 1 : v >= 4 ? 0 : (4 - v) / 2;
}
function t0Mw(v: number): number {
  return v <= 360 ? 1 : v >= 500 ? 0 : (500 - v) / 140;
}
function t0Tpsa(v: number): number {
  if (v <= 20 || v >= 120) return 0;
  if (v >= 40 && v <= 90) return 1;
  return v < 40 ? (v - 20) / 20 : (120 - v) / 30;
}
function t0Hbd(v: number): number {
  return v <= 0.5 ? 1 : v >= 3.5 ? 0 : (3.5 - v) / 3;
}
function t0Pka(v: number): number {
  return v <= 8 ? 1 : v >= 10 ? 0 : (10 - v) / 2;
}

function formalChargeFromSmiles(smi: string): number {
  let q = 0;
  const matches = smi.match(/\[[^\]]+\]/g);
  if (!matches) return 0;
  for (const tok of matches) {
    const plusN = tok.match(/\+(\d+)/);
    const minusN = tok.match(/-(\d+)/);
    if (plusN) q += Number.parseInt(plusN[1], 10);
    else if (tok.includes('+')) q += (tok.match(/\+/g) || []).length;
    if (minusN) q -= Number.parseInt(minusN[1], 10);
    else if (tok.includes('-')) q -= (tok.match(/-/g) || []).length;
  }
  return q;
}

function drawThemedSvg(mol: RDKitModule, theme: ThemeMode): string {
  const details =
    theme === 'dark'
      ? {
          width: 280,
          height: 165,
          bondLineWidth: 2,
          clearBackground: false,
          atomColourPalette: {
            0: [0.89, 0.93, 0.98],
            6: [0.89, 0.93, 0.98],
            7: [0.38, 0.65, 0.98],
            8: [0.98, 0.44, 0.44],
            9: [0.29, 0.87, 0.5],
            15: [0.98, 0.57, 0.24],
            16: [0.98, 0.8, 0.08],
            17: [0.2, 0.83, 0.6],
            35: [0.75, 0.52, 0.99],
            53: [0.85, 0.47, 0.94],
          },
        }
      : {
          width: 280,
          height: 165,
          bondLineWidth: 2,
          clearBackground: false,
        };
  try {
    return String(mol.get_svg_with_highlights(JSON.stringify(details)))
      .replace(/width='280px' height='165px'/, "viewBox='0 0 280 165' width='100%' height='100%'");
  } catch {
    return String(mol.get_svg(280, 165));
  }
}

export class BrowserChemScorer {
  private rdkit: RDKitModule;
  private admetEndpoints: Record<'bbb' | 'pgp' | 'herg', DecodedEndpoint>;
  private qsarEndpoints: Record<string, DecodedEndpoint>;
  private basicQmol: { qmol: RDKitModule; pka: number }[];
  private acidicQmol: { qmol: RDKitModule; pka: number }[];

  constructor(rdkit: RDKitModule, admet: BrowserAdmetPack, qsar: BrowserQsarPack | null) {
    this.rdkit = rdkit;
    this.admetEndpoints = {
      bbb: decodeEndpoint(admet.endpoints.bbb),
      pgp: decodeEndpoint(admet.endpoints.pgp),
      herg: decodeEndpoint(admet.endpoints.herg),
    };
    this.qsarEndpoints = {};
    if (qsar?.endpoints) {
      for (const [tgt, ep] of Object.entries(qsar.endpoints)) {
        this.qsarEndpoints[tgt] = decodeEndpoint(ep);
      }
    }
    this.basicQmol = [];
    for (const r of admet.pka_smarts_rules?.basic ?? []) {
      const qm = rdkit.get_qmol(r.smarts);
      if (qm) this.basicQmol.push({ qmol: qm, pka: r.pka });
    }
    this.acidicQmol = [];
    for (const r of admet.pka_smarts_rules?.acidic ?? []) {
      const qm = rdkit.get_qmol(r.smarts);
      if (qm) this.acidicQmol.push({ qmol: qm, pka: r.pka });
    }
  }

  score(rawSmiles: string): LiveMoleculeScore | null {
    const clean = rawSmiles.trim();
    if (!clean) return null;
    const mol = this.rdkit.get_mol(clean);
    if (!mol || !mol.is_valid()) {
      mol?.delete();
      return null;
    }

    try {
      const canonicalSmiles = String(mol.get_smiles());
      const desc = JSON.parse(String(mol.get_descriptors())) as Record<string, number>;
      const fpBits = String(mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 1024 })));

      let basicPka = 2.0;
      for (const { qmol, pka } of this.basicQmol) {
        const match = String(mol.get_substruct_match(qmol));
        if (match && match !== '{}' && pka > basicPka) basicPka = pka;
      }
      let acidicPka: number | null = null;
      for (const { qmol, pka } of this.acidicQmol) {
        const match = String(mol.get_substruct_match(qmol));
        if (match && match !== '{}' && (acidicPka === null || pka < acidicPka)) {
          acidicPka = pka;
        }
      }

      const mw = Number(desc.amw ?? desc.exactmw ?? 0);
      const clogp = Number(desc.CrippenClogP ?? 0);
      const tpsa = Number(desc.tpsa ?? 0);
      const hbd = Number(desc.NumHBD ?? 0);
      const hba = Number(desc.NumHBA ?? 0);
      const rotBonds = Number(desc.NumRotatableBonds ?? 0);
      const fracCsp3 = Number(desc.FractionCSP3 ?? 0);
      const ringCount = Number(desc.NumRings ?? 0);
      const aromRings = Number(desc.NumAromaticRings ?? 0);
      const heavyAtoms = Number(desc.NumHeavyAtoms ?? 0);
      const formalCharge = formalChargeFromSmiles(canonicalSmiles);

      // Henderson-Hasselbalch ClogD_7.4 matching src/geneuclidean/cns/mpo.py
      let clogd74 = clogp;
      if (basicPka >= 13.5) {
        clogd74 = clogp - 3.5;
      } else {
        const baseRatio = basicPka > 2.5 ? 10 ** (basicPka - 7.4) : 0;
        const acidRatio = acidicPka !== null ? 10 ** (7.4 - acidicPka) : 0;
        clogd74 = clogp - Math.min(Math.log10(1 + baseRatio + acidRatio), 4.0);
      }

      const cnsMpoV1 = Math.max(
        0,
        Math.min(
          6,
          t0ClogP(clogp) + t0ClogD(clogd74) + t0Mw(mw) + t0Tpsa(tpsa) + t0Hbd(hbd) + t0Pka(basicPka),
        ),
      );
      const cnsMpoV2 = Math.max(
        0,
        Math.min(
          6,
          t0ClogP(clogp) + t0Mw(mw) + t0Tpsa(tpsa) + t0Pka(basicPka) + 2 * t0Hbd(hbd),
        ),
      );

      // 1036-dim shared feature vector
      const feat = new Float32Array(1036);
      const fpLen = Math.min(1024, fpBits.length);
      for (let i = 0; i < fpLen; i++) feat[i] = fpBits.charCodeAt(i) === 49 ? 1 : 0;
      feat[1024] = mw;
      feat[1025] = clogp;
      feat[1026] = tpsa;
      feat[1027] = hbd;
      feat[1028] = hba;
      feat[1029] = rotBonds;
      feat[1030] = fracCsp3;
      feat[1031] = ringCount;
      feat[1032] = aromRings;
      feat[1033] = heavyAtoms;
      feat[1034] = formalCharge;
      feat[1035] = basicPka;

      const bbbProb = runEndpoint(this.admetEndpoints.bbb, feat);
      const pgpProb = runEndpoint(this.admetEndpoints.pgp, feat);
      const hergProb = runEndpoint(this.admetEndpoints.herg, feat);

      const targetProbs: Record<string, number> = {};
      for (const [tgt, ep] of Object.entries(this.qsarEndpoints)) {
        targetProbs[tgt] = runEndpoint(ep, feat);
      }

      const jointCnsSafe = cnsMpoV2 >= 4.0 && bbbProb >= 0.65 && pgpProb <= 0.35 && hergProb <= 0.3;

      return {
        valid: true,
        canonicalSmiles,
        svgDark: drawThemedSvg(mol, 'dark'),
        svgLight: drawThemedSvg(mol, 'light'),
        mw,
        clogp,
        clogd74,
        tpsa,
        hbd,
        hba,
        rotBonds,
        mostBasicPka: basicPka,
        cnsMpoV1,
        cnsMpoV2,
        bbbProb,
        pgpProb,
        hergProb,
        jointCnsSafe,
        targetProbs,
      };
    } finally {
      mol.delete();
    }
  }
}

let cachedScorer: Promise<BrowserChemScorer> | null = null;

export function loadBrowserChemScorer(): Promise<BrowserChemScorer> {
  if (cachedScorer) return cachedScorer;
  cachedScorer = (async () => {
    const [rdkit, admetRes, qsarRes] = await Promise.all([
      loadRDKit(),
      fetch(assetUrl('models/cns_admet_browser.json')),
      fetch(assetUrl('models/cns_target_qsar_browser.json')).catch(() => null),
    ]);
    if (!admetRes.ok) {
      throw new Error('Browser ADMET oracle JSON not found in web/public/models/.');
    }
    const admet = (await admetRes.json()) as BrowserAdmetPack;
    const qsar = qsarRes && qsarRes.ok ? ((await qsarRes.json()) as BrowserQsarPack) : null;
    return new BrowserChemScorer(rdkit, admet, qsar);
  })();
  cachedScorer.catch(() => {
    cachedScorer = null;
  });
  return cachedScorer;
}
