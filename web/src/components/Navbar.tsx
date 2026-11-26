import {
  Atom,
  BarChart3,
  BookOpen,
  Brain,
  Cpu,
  FlaskConical,
  Github,
  Layers,
  Moon,
  Sparkles,
  Sun,
  Workflow,
} from 'lucide-react';
import type { StudioTab, ThemeMode } from '../types/studio';

interface NavbarProps {
  activeTab: StudioTab;
  onSelectTab: (tab: StudioTab) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  version: string;
  repoUrl: string;
}

const NAV_ITEMS: { id: StudioTab; label: string; shortLabel: string; icon: typeof Layers; badge?: string }[] = [
  { id: 'overview', label: 'Overview', shortLabel: 'Overview', icon: Layers },
  { id: 'studio3d', label: '3D & 2D Studio', shortLabel: '3D Studio', icon: Atom },
  { id: 'circuit', label: 'Circuit Studio', shortLabel: 'Circuit', icon: Workflow, badge: '49' },
  { id: 'generator', label: 'Live AI Generator', shortLabel: 'AI Gen', icon: Sparkles, badge: 'AI' },
  { id: 'experiments', label: 'Experiments', shortLabel: 'Benchmarks', icon: BarChart3 },
  { id: 'disorders', label: 'Disorder Portals', shortLabel: 'Disorders', icon: Brain },
  { id: 'figures', label: 'Figures & Docs', shortLabel: 'Figures', icon: BookOpen },
];

export function Navbar({
  activeTab,
  onSelectTab,
  theme,
  onToggleTheme,
  version,
  repoUrl,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 glass-panel border-b border-[var(--border-subtle)]">
      <div className="mx-auto max-w-[1600px] px-3 sm:px-5 h-14 flex items-center justify-between gap-2">
        {/* Brand mark */}
        <button
          type="button"
          onClick={() => onSelectTab('overview')}
          className="flex items-center gap-2.5 group shrink-0 text-left cursor-pointer"
        >
          <div className="relative h-8 w-8 rounded-lg bg-gradient-to-br from-cyan-500 via-sky-500 to-emerald-500 p-[1px] shadow-sm shadow-cyan-500/20">
            <div className="h-full w-full rounded-[7px] bg-slate-950 dark:bg-[#060913] flex items-center justify-center">
              <Cpu className="h-4 w-4 text-cyan-400 group-hover:scale-110 transition-transform" />
            </div>
          </div>
          <div className="leading-none">
            <div className="flex items-center gap-1.5">
              <span className="font-bold tracking-tight text-sm sm:text-[15px] text-[var(--text-primary)]">
                GenEuclidean
              </span>
              <span className="font-mono-num text-[10px] font-semibold px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-600 dark:text-cyan-300 border border-cyan-500/25">
                {version}
              </span>
            </div>
            <span className="hidden xl:block text-[10px] text-[var(--text-muted)] tracking-wide mt-0.5">
              49-Block 3D SE(3) · NLP · NanoFold3 Studio
            </span>
          </div>
        </button>

        {/* Primary 6-view studio navigation */}
        <nav className="flex items-center gap-0.5 sm:gap-1 overflow-x-auto no-scrollbar py-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`relative flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                  active
                    ? 'bg-cyan-500/15 text-cyan-700 dark:text-cyan-200 border border-cyan-500/35 shadow-sm'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-slate-500/10 border border-transparent'
                }`}
              >
                <Icon className="h-3.5 w-3.5 shrink-0" />
                <span className="hidden md:inline">{item.label}</span>
                <span className="md:hidden">{item.shortLabel}</span>
                {item.badge && (
                  <span className="ml-0.5 text-[9px] font-mono-num font-bold uppercase px-1 py-0.2 rounded bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 border border-emerald-500/30">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Utility actions */}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          <span
            title="All molecules are in silico computational hypotheses for academic drug-discovery research, not clinically tested medicines."
            className="hidden 2xl:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold bg-amber-500/10 text-amber-700 dark:text-amber-300 border border-amber-500/25"
          >
            <FlaskConical className="h-3 w-3" />
            In Silico Research
          </span>

          <button
            type="button"
            onClick={onToggleTheme}
            title={theme === 'dark' ? 'Switch to Publication Paper Light Mode' : 'Switch to Obsidian Lab Dark Mode'}
            className="h-8 w-8 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] hover:border-cyan-500/40 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors cursor-pointer"
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          <a
            href={repoUrl}
            target="_blank"
            rel="noreferrer"
            title="View GenEuclidean on GitHub"
            className="h-8 px-2.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-surface)] hover:border-cyan-500/40 flex items-center gap-1.5 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
          >
            <Github className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
