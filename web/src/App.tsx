import { lazy, Suspense, useEffect, useLayoutEffect, useState } from 'react';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Navbar } from './components/Navbar';
import { loadExperiments, loadMoleculesIndex, loadOverview, loadV4Benchmarks } from './lib/data';
import { studioTabs, type StudioTab, type ThemeMode } from './types/studio';

const HumanTwin = lazy(() => import('./views/HumanTwinView').then(m => ({ default: m.HumanTwinView })));
const Overview = lazy(() => import('./views/OverviewView').then(m => ({ default: m.OverviewView })));
const Studio3D = lazy(() => import('./views/Studio3DView').then(m => ({ default: m.Studio3DView })));
const Circuit = lazy(() => import('./views/CircuitBuilderView').then(m => ({ default: m.CircuitBuilderView })));
const Generator = lazy(() => import('./views/GeneratorView').then(m => ({ default: m.GeneratorView })));
const Experiments = lazy(() => import('./views/ExperimentsView').then(m => ({ default: m.ExperimentsView })));
const Disorders = lazy(() => import('./views/DisordersView').then(m => ({ default: m.DisordersView })));
const Figures = lazy(() => import('./views/FiguresView').then(m => ({ default: m.FiguresView })));

const titles: Record<StudioTab, string> = {
  overview: 'Molecular studio', 'human-twin': 'Tourette simulator', studio3d: 'Structure viewer',
  circuit: 'Circuit studio', generator: 'Molecule generator', experiments: 'Experiments',
  disorders: 'Disorder portals', figures: 'Figures and documentation',
};
const repoUrl = 'https://github.com/daniil-777/geneuclidean';

function currentTab(): StudioTab {
  const route = window.location.hash.slice(1).split('?')[0];
  return studioTabs.find(tab => tab === route) ?? 'overview';
}

function useDataset<T>(load: () => Promise<T>, enabled: boolean, attempt: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled || data) return;
    let active = true;
    setError(null);
    load().then(value => { if (active) setData(value); }).catch((reason: unknown) => {
      if (active) setError(reason instanceof Error ? reason.message : 'The dataset could not be loaded.');
    });
    return () => { active = false; };
  }, [load, enabled, data, attempt]);
  return { data, error: enabled ? error : null, loading: enabled && !data };
}

function LoadingView() {
  return <div className="studio-loading" role="status">Opening workspace…</div>;
}

export function App() {
  const [activeTab, setActiveTab] = useState(currentTab);
  const [theme, setTheme] = useState<ThemeMode>(() =>
    document.documentElement.classList.contains('dark') ? 'dark' : 'light');
  const [selectedTarget, setSelectedTarget] = useState('DRD1');
  const [disorderFilter, setDisorderFilter] = useState('All');
  const [attempt, setAttempt] = useState(0);
  const overview = useDataset(loadOverview,
    ['overview', 'experiments', 'disorders', 'figures'].includes(activeTab), attempt);
  const molecules = useDataset(loadMoleculesIndex,
    ['studio3d', 'circuit', 'generator', 'disorders'].includes(activeTab), attempt);
  const experiments = useDataset(loadExperiments, activeTab === 'experiments', attempt);
  const benchmarks = useDataset(loadV4Benchmarks, ['circuit', 'experiments'].includes(activeTab), attempt);

  useLayoutEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    try { window.localStorage.setItem('studio-theme', theme); } catch { /* storage may be disabled */ }
  }, [theme]);

  useEffect(() => {
    function routeChanged() {
      if (currentTab() !== activeTab) {
        setActiveTab(currentTab());
        window.scrollTo(0, 0);
        document.getElementById('studio-main')?.focus({ preventScroll: true });
      }
    }
    window.addEventListener('hashchange', routeChanged);
    return () => window.removeEventListener('hashchange', routeChanged);
  }, [activeTab]);

  useEffect(() => { document.title = titles[activeTab] + ' · GenEuclidean'; }, [activeTab]);

  function navigate(tab: StudioTab, target?: string, disorder?: string) {
    if (target) setSelectedTarget(target);
    if (disorder !== undefined) setDisorderFilter(disorder);
    if (tab !== activeTab) window.location.hash = tab;
    else window.scrollTo(0, 0);
  }

  const required = [overview, molecules, experiments];
  const error = required.find(resource => resource.error)?.error;
  let view;
  if (error) {
    view = <div className="studio-loading" role="alert">
      <p>This workspace’s dataset could not be loaded.</p>
      <button type="button" onClick={() => setAttempt(n => n + 1)}>Try again</button>
    </div>;
  } else if (required.some(resource => resource.loading)) {
    view = <LoadingView />;
  } else {
    switch (activeTab) {
      case 'human-twin': view = <HumanTwin />; break;
      case 'overview': view = overview.data && <Overview overview={overview.data} theme={theme}
        onNavigate={(tab, target) => navigate(tab, target, 'All')} />; break;
      case 'studio3d': view = molecules.data && <Studio3D molecules={molecules.data.molecules}
        initialTarget={selectedTarget} initialDisorderFilter={disorderFilter} theme={theme}
        onSelectTarget={setSelectedTarget} />; break;
      case 'circuit': view = molecules.data && <Circuit molecules={molecules.data.molecules}
        v4Benchmarks={benchmarks.data} theme={theme} onInspectIn3D={target => navigate('studio3d', target, 'All')} />; break;
      case 'generator': view = molecules.data && <Generator molecules={molecules.data.molecules}
        theme={theme} onInspectIn3D={target => navigate('studio3d', target, 'All')} />; break;
      case 'experiments': view = experiments.data && overview.data && <Experiments experiments={experiments.data}
        v4Benchmarks={benchmarks.data} casfModels={overview.data.model_comparison_casf}
        theme={theme} onInspectTarget={target => navigate('studio3d', target, 'All')} />; break;
      case 'disorders': view = molecules.data && overview.data && <Disorders molecules={molecules.data.molecules}
        v4Discovery={overview.data.v4_discovery} onLaunchStudio={(target, disorder) => navigate('studio3d', target, disorder)} />; break;
      case 'figures': view = overview.data && <Figures figures={overview.data.figures} repoUrl={overview.data.repo_url} />; break;
    }
  }
  return (
    <div className="studio-app min-h-screen flex flex-col">
      <button className="studio-skip" onClick={() => document.getElementById('studio-main')?.focus()}>Skip to workspace</button>
      <Navbar activeTab={activeTab} onSelectTab={navigate} theme={theme}
        onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
        version={overview.data?.version ?? 'v5.3.0'} repoUrl={overview.data?.repo_url ?? repoUrl} />
      <main id="studio-main" tabIndex={-1} className="flex-1 min-w-0">
        <ErrorBoundary key={activeTab} fallbackTitle="This workspace could not open">
          <Suspense fallback={<LoadingView />}>{view}</Suspense>
        </ErrorBoundary>
      </main>
      {activeTab !== 'circuit' && <footer className="studio-footer">
        <span>GenEuclidean · Molecular research studio</span>
        <a href={repoUrl}>Source code ↗</a>
      </footer>}
    </div>
  );
}
export default App;
