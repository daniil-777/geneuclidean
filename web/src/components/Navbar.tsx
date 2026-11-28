import {
  ArrowUpRight, Box, ChevronDown, GitFork, Menu, Moon, Orbit, Sun, X,
} from 'lucide-react';
import { useEffect, useRef, useState, type MouseEvent } from 'react';
import type { StudioTab, ThemeMode } from '../types/studio';

interface NavbarProps {
  activeTab: StudioTab;
  onSelectTab: (tab: StudioTab) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  version: string;
  repoUrl: string;
}

const PRIMARY: { id: StudioTab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'human-twin', label: 'Human twin' },
  { id: 'studio3d', label: '3D viewer' },
];
const TOOLS: typeof PRIMARY = [
  { id: 'circuit', label: 'Circuit builder' },
  { id: 'generator', label: 'Molecule generator' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'disorders', label: 'Disorders' },
  { id: 'figures', label: 'Figures & docs' },
];

export function Navbar({ activeTab, onSelectTab, theme, onToggleTheme, version, repoUrl }: NavbarProps) {
  const [menu, setMenu] = useState<'desktop' | 'mobile' | null>(null);
  const header = useRef<HTMLElement>(null);
  const desktopTrigger = useRef<HTMLButtonElement>(null);
  const mobileTrigger = useRef<HTMLButtonElement>(null);

  useEffect(() => { setMenu(null); }, [activeTab]);
  useEffect(() => {
    if (!menu) return;
    const clickAway = (e: PointerEvent) => {
      if (!header.current?.contains(e.target as Node)) setMenu(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      setMenu(null);
      (menu === 'desktop' ? desktopTrigger : mobileTrigger).current?.focus();
    };
    const onLayoutChange = () => setMenu(null);
    const breakpoint = matchMedia('(max-width: 880px)');
    document.addEventListener('pointerdown', clickAway);
    document.addEventListener('keydown', onKey);
    breakpoint.addEventListener('change', onLayoutChange);
    return () => {
      document.removeEventListener('pointerdown', clickAway);
      document.removeEventListener('keydown', onKey);
      breakpoint.removeEventListener('change', onLayoutChange);
    };
  }, [menu]);

  function choose(e: MouseEvent<HTMLAnchorElement>, id: StudioTab) {
    if (e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    setMenu(null);
    onSelectTab(id);
  }
  function link({ id, label }: typeof PRIMARY[number]) {
    return (
      <a key={id} href={`#${id}`} onClick={e => choose(e, id)}
        aria-current={activeTab === id ? 'page' : undefined} className="studio-nav-link">
        {label}
      </a>
    );
  }

  return (
    <header ref={header} className="studio-header" onBlur={e => {
      if (e.relatedTarget && !e.currentTarget.contains(e.relatedTarget)) setMenu(null);
    }}>
      <div className="studio-nav-inner">
        <a href="#overview" className="studio-brand" onClick={e => choose(e, 'overview')} aria-label="GenEuclidean overview">
          <span className="studio-mark" aria-hidden="true"><Orbit size={24} strokeWidth={1.75}/></span>
          GenEuclidean
        </a>
        <nav className="studio-primary-nav" aria-label="Studio navigation">
          {PRIMARY.map(link)}
          <div className="studio-tools-wrap">
            <button type="button" ref={desktopTrigger} aria-expanded={menu === 'desktop'}
              aria-controls="studio-tools" className={`studio-nav-link studio-tools-trigger ${TOOLS.some(item => item.id === activeTab) ? 'current-group' : ''}`}
              onClick={() => setMenu(menu === 'desktop' ? null : 'desktop')}>
              Tools <ChevronDown size={14}/>
            </button>
            {menu === 'desktop' && <div id="studio-tools" className="studio-drop">{TOOLS.map(link)}</div>}
          </div>
        </nav>
        <div className="studio-utilities">
          <a href={repoUrl} target="_blank" rel="noreferrer" className="studio-repo" aria-label="View source on GitHub (opens in a new tab)">
            <GitFork size={15} aria-hidden="true"/><span>Source</span><ArrowUpRight size={13} aria-hidden="true"/>
          </a>
          <button type="button" className="studio-icon-button" onClick={onToggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}
          </button>
          <button type="button" ref={mobileTrigger} className="studio-icon-button studio-menu-toggle" aria-controls="studio-mobile-nav"
            aria-expanded={menu === 'mobile'} aria-label={menu === 'mobile' ? 'Close navigation' : 'Open navigation'}
            onClick={() => setMenu(menu === 'mobile' ? null : 'mobile')}>
            {menu === 'mobile' ? <X size={21}/> : <Menu size={21}/>}
          </button>
        </div>
      </div>
      {menu === 'mobile' &&
        <nav id="studio-mobile-nav" className="studio-mobile-nav" aria-label="Studio navigation">
          {[...PRIMARY, ...TOOLS].map(link)}
          <div className="studio-mobile-meta"><Box size={14}/> Studio {version}</div>
        </nav>}
    </header>
  );
}
