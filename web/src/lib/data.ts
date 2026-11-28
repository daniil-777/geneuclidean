import type {
  ComplexDetail,
  ExperimentsData,
  MoleculesIndexData,
  OverviewData,
  V4BenchmarksData,
} from '../types/studio';

const BASE = import.meta.env.BASE_URL.endsWith('/')
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`;

export function assetUrl(relPath: string): string {
  const clean = relPath.replace(/^\/+/, '');
  return `${BASE}${clean}`;
}

export function figureUrl(filename: string): string {
  return assetUrl(`figures/${filename}`);
}

const jsonCache = new Map<string, Promise<unknown>>();

function fetchJsonCached<T>(relPath: string): Promise<T> {
  const url = assetUrl(relPath);
  const existing = jsonCache.get(url);
  if (existing) {
    return existing as Promise<T>;
  }
  const promise = fetch(url).then((res) => {
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} loading ${url}`);
    }
    return res.json() as Promise<T>;
  }).catch((error: unknown) => {
    jsonCache.delete(url);
    throw error;
  });
  jsonCache.set(url, promise);
  return promise;
}

export function loadOverview(): Promise<OverviewData> {
  return fetchJsonCached<OverviewData>('data/overview.json');
}

export function loadMoleculesIndex(): Promise<MoleculesIndexData> {
  return fetchJsonCached<MoleculesIndexData>('data/molecules_index.json');
}

export function loadExperiments(): Promise<ExperimentsData> {
  return fetchJsonCached<ExperimentsData>('data/experiments.json');
}

export function loadV4Benchmarks(): Promise<V4BenchmarksData> {
  return fetchJsonCached<V4BenchmarksData>('data/v4_benchmarks.json');
}

export function loadComplexDetail(target: string): Promise<ComplexDetail> {
  return fetchJsonCached<ComplexDetail>(`data/complexes/${encodeURIComponent(target)}.json`);
}

export function formatPValue(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return '—';
  if (p < 0.001) {
    const exp = Math.floor(Math.log10(Math.max(p, 1e-300)));
    const mant = p / 10 ** exp;
    return `${mant.toFixed(1)}×10^${exp}`;
  }
  if (p < 0.01) return p.toFixed(4);
  if (p < 0.05) return p.toFixed(3);
  return p.toFixed(2);
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function downloadTextFile(filename: string, content: string, mime = 'text/plain'): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
