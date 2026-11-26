import { useEffect, useState } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Navbar } from './components/Navbar';
import {
  loadExperiments,
  loadMoleculesIndex,
  loadOverview,
  loadV4Benchmarks,
} from './lib/data';
import type {
  ExperimentsData,
  MoleculesIndexData,
  OverviewData,
  StudioTab,
  ThemeMode,
  V4BenchmarksData,
} from './types/studio';
import { CircuitBuilderView } from './views/CircuitBuilderView';
import { DisordersView } from './views/DisordersView';
import { ExperimentsView } from './views/ExperimentsView';
import { FiguresView } from './views/FiguresView';
import { GeneratorView } from './views/GeneratorView';
import { OverviewView } from './views/OverviewView';
import { Studio3DView } from './views/Studio3DView';

export function App() {
  const [activeTab, setActiveTab] = useState<StudioTab>('overview');
  const [theme, setTheme] = useState<ThemeMode>('dark');
  const [selectedTarget, setSelectedTarget] = useState<string>('DRD1');
  const [disorderFilter, setDisorderFilter] = useState<string>('All');

  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [moleculesIdx, setMoleculesIdx] = useState<MoleculesIndexData | null>(null);
  const [experiments, setExperiments] = useState<ExperimentsData | null>(null);
  const [v4Benchmarks, setV4Benchmarks] = useState<V4BenchmarksData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Apply dark/light root class
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  // Initial data load (overview + molecules_index); lazy-load experiments on demand or background
  useEffect(() => {
    let alive = true;
    Promise.all([loadOverview(), loadMoleculesIndex()])
      .then(([ov, mols]) => {
        if (!alive) return;
        setOverview(ov);
        setMoleculesIdx(mols);
      })
      .catch((err) => {
        if (!alive) return;
        setLoadError(err instanceof Error ? err.message : 'Failed to load studio datasets.');
      });
    return () => {
      alive = false;
    };
  }, []);

  // Load experiments & v4 benchmarks JSONs after overview mounts
  useEffect(() => {
    if ((experiments && v4Benchmarks) || !overview) return;
    let alive = true;
    if (!experiments) {
      loadExperiments()
        .then((exp) => {
          if (alive) setExperiments(exp);
        })
        .catch(() => {
          // retry when clicking experiments tab
        });
    }
    if (!v4Benchmarks) {
      loadV4Benchmarks()
        .then((v4) => {
          if (alive) setV4Benchmarks(v4);
        })
        .catch(() => {
          // optional v4 payload
        });
    }
    return () => {
      alive = false;
    };
  }, [overview, experiments, v4Benchmarks]);

  const handleNavigate = (tab: StudioTab, target?: string, disorder?: string) => {
    if (target) setSelectedTarget(target);
    if (disorder !== undefined) setDisorderFilter(disorder);
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (loadError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="lab-card rounded-2xl max-w-md w-full p-6 text-center space-y-3">
          <div className="text-rose-500 font-bold text-base">Failed to Load Studio Data</div>
          <p className="text-xs text-[var(--text-secondary)] font-mono-num">{loadError}</p>
          <p className="text-xs text-[var(--text-muted)]">
            Run <code>.venv/bin/python scripts/build_web_data.py</code> to compile verified JSON
            assets under <code>web/public/data/</code>.
          </p>
        </div>
      </div>
    );
  }

  if (!overview || !moleculesIdx) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-3">
        <div className="h-8 w-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
        <div className="font-mono-num text-xs text-[var(--text-muted)]">
          Loading GenEuclidean Scientific Studio…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar
        activeTab={activeTab}
        onSelectTab={(tab) => handleNavigate(tab)}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        version={overview.version}
        repoUrl={overview.repo_url}
      />

      <main className="flex-1">
        <ErrorBoundary key={activeTab} fallbackTitle="GenEuclidean Studio View Error">
          {activeTab === 'overview' && (
            <OverviewView
              overview={overview}
              theme={theme}
              onNavigate={(tab, target) => handleNavigate(tab, target, 'All')}
            />
          )}

          {activeTab === 'studio3d' && (
            <Studio3DView
              molecules={moleculesIdx.molecules}
              initialTarget={selectedTarget}
              initialDisorderFilter={disorderFilter}
              theme={theme}
              onSelectTarget={setSelectedTarget}
            />
          )}

          {activeTab === 'circuit' && (
            <CircuitBuilderView
              molecules={moleculesIdx.molecules}
              v4Benchmarks={v4Benchmarks}
              theme={theme}
              onInspectIn3D={(target) => handleNavigate('studio3d', target, 'All')}
            />
          )}

          {activeTab === 'generator' && (
            <GeneratorView
              molecules={moleculesIdx.molecules}
              theme={theme}
              onInspectIn3D={(target) => handleNavigate('studio3d', target, 'All')}
            />
          )}

          {activeTab === 'experiments' &&
            (experiments ? (
              <ExperimentsView
                experiments={experiments}
                v4Benchmarks={v4Benchmarks}
                casfModels={overview.model_comparison_casf}
                theme={theme}
                onInspectTarget={(target) => handleNavigate('studio3d', target, 'All')}
              />
            ) : (
              <div className="min-h-[60vh] flex flex-col items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
                <div className="h-6 w-6 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin" />
                Loading interactive experimental datasets…
              </div>
            ))}

          {activeTab === 'disorders' && (
            <DisordersView
              molecules={moleculesIdx.molecules}
              v4Discovery={overview.v4_discovery}
              onLaunchStudio={(target, disorder) => handleNavigate('studio3d', target, disorder)}
            />
          )}

          {activeTab === 'figures' && (
            <FiguresView figures={overview.figures} repoUrl={overview.repo_url} />
          )}
        </ErrorBoundary>
      </main>

      {activeTab !== 'circuit' && (
        <footer className="border-t border-[var(--border-subtle)] py-4 px-5 mt-8">
          <div className="mx-auto max-w-[1600px] flex flex-wrap items-center justify-between gap-2 text-[11px] text-[var(--text-muted)]">
            <div>
              <strong className="text-[var(--text-secondary)]">GenEuclidean {overview.version}</strong>{' '}
              · SE(3)-Equivariant 3D Pocket-Conditioned Molecular Studio · MIT License
            </div>
            <div>
              Strictly <em>in silico</em> computational research hypotheses — not clinically tested
              medicines.
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}

export default App;
