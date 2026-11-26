import {
  ChevronDown,
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  X,
} from 'lucide-react';
import type React from 'react';
import { useMemo, useState } from 'react';
import type { BlockCategoryId, BlockDefinition } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import {
  BLOCK_CATALOG,
  BLOCK_CATEGORIES,
  getCategoryColor,
  getSignalColor,
} from '../../lib/circuitCatalog';
import { CATEGORY_ICONS } from './categoryIcons';

interface BlockPaletteProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onAddBlock: (blockType: string) => void;
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  theme: ThemeMode;
}

export function BlockPalette({
  collapsed,
  onToggleCollapse,
  onAddBlock,
  searchInputRef,
  theme,
}: BlockPaletteProps) {
  const [query, setQuery] = useState('');
  const [openCategories, setOpenCategories] = useState<Record<BlockCategoryId, boolean>>({
    inputs: true,
    encoders: true,
    generators: true,
    screening: false,
    active_learning: true,
    physics_audit: false,
    leadopt_admet: false,
    outputs: false,
  });

  const filteredByCategory = useMemo(() => {
    const q = query.trim().toLowerCase();
    const grouped = new Map<BlockCategoryId, BlockDefinition[]>();
    for (const cat of BLOCK_CATEGORIES) grouped.set(cat.id, []);
    for (const block of BLOCK_CATALOG) {
      if (
        !q ||
        block.name.toLowerCase().includes(q) ||
        block.shortName.toLowerCase().includes(q) ||
        block.blockType.toLowerCase().includes(q) ||
        block.summary.toLowerCase().includes(q) ||
        block.badge.toLowerCase().includes(q) ||
        block.pythonSymbol.toLowerCase().includes(q)
      ) {
        grouped.get(block.category)?.push(block);
      }
    }
    return grouped;
  }, [query]);

  const totalMatches = useMemo(() => {
    let n = 0;
    filteredByCategory.forEach((list) => {
      n += list.length;
    });
    return n;
  }, [filteredByCategory]);

  if (collapsed) {
    return (
      <aside className="w-11 shrink-0 h-full circuit-glass border-r border-[var(--circuit-node-border)] flex flex-col items-center py-2.5 gap-2 z-20">
        <button
          type="button"
          onClick={onToggleCollapse}
          title="Expand Block Dock"
          className="h-8 w-8 rounded-lg hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <PanelLeftOpen className="h-4 w-4" />
        </button>
        <div className="w-6 h-[1px] bg-[var(--circuit-node-border)]" />
        <div className="flex flex-col items-center gap-1.5 overflow-y-auto no-scrollbar">
          {BLOCK_CATEGORIES.map((cat) => {
            const Icon = CATEGORY_ICONS[cat.id];
            const color = getCategoryColor(cat.id, theme);
            return (
              <button
                key={cat.id}
                type="button"
                title={cat.label}
                onClick={() => {
                  onToggleCollapse();
                  setOpenCategories((prev) => ({ ...prev, [cat.id]: true }));
                }}
                style={{ color }}
                className="h-7 w-7 rounded-lg hover:bg-slate-500/12 flex items-center justify-center cursor-pointer"
              >
                <Icon className="h-3.5 w-3.5" />
              </button>
            );
          })}
        </div>
      </aside>
    );
  }

  return (
    <aside
      data-block-palette="true"
      className="w-[254px] shrink-0 h-full circuit-glass border-r border-[var(--circuit-node-border)] flex flex-col z-20"
    >
      {/* Dock Header */}
      <div className="px-3 pt-2.5 pb-2 border-b border-[var(--circuit-node-border)] space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold tracking-tight text-[var(--text-primary)]">
              Block Dock
            </span>
            <span className="font-mono-num text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-cyan-500/12 text-cyan-700 dark:text-cyan-300">
              {totalMatches}
            </span>
          </div>
          <button
            type="button"
            onClick={onToggleCollapse}
            title="Collapse Block Dock"
            className="h-6 w-6 rounded-md hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Search Input with ⌘K hint */}
        <div className="relative">
          <Search className="h-3.5 w-3.5 text-[var(--text-muted)] absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            ref={searchInputRef}
            type="text"
            value={query}
            onChange={(ev) => setQuery(ev.target.value)}
            placeholder="Search 36 blocks… (⌘K)"
            className="w-full h-8 pl-8 pr-7 rounded-lg text-[11.5px] bg-[var(--bg-elevated)]/85 border border-[var(--circuit-node-border)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-cyan-500/60"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {/* Category Accordions */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1.5">
        {BLOCK_CATEGORIES.map((cat) => {
          const blocks = filteredByCategory.get(cat.id) ?? [];
          if (blocks.length === 0) return null;
          const isOpen = query.trim().length > 0 || openCategories[cat.id];
          const Icon = CATEGORY_ICONS[cat.id];
          const color = getCategoryColor(cat.id, theme);

          return (
            <div key={cat.id} className="space-y-1">
              <button
                type="button"
                onClick={() =>
                  setOpenCategories((prev) => ({ ...prev, [cat.id]: !prev[cat.id] }))
                }
                className="w-full px-1.5 py-1 rounded-lg flex items-center justify-between gap-1.5 text-left hover:bg-slate-500/10 transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  {isOpen ? (
                    <ChevronDown className="h-3 w-3 text-[var(--text-muted)] shrink-0" />
                  ) : (
                    <ChevronRight className="h-3 w-3 text-[var(--text-muted)] shrink-0" />
                  )}
                  <Icon className="h-3.5 w-3.5 shrink-0" style={{ color }} />
                  <span className="text-[11px] font-semibold text-[var(--text-primary)] truncate">
                    {cat.shortLabel}
                  </span>
                </div>
                <span className="font-mono-num text-[10px] text-[var(--text-muted)] shrink-0">
                  {blocks.length}
                </span>
              </button>

              {isOpen && (
                <div className="space-y-1 pl-1">
                  {blocks.map((block) => (
                    <div
                      key={block.blockType}
                      draggable
                      data-palette-block={block.blockType}
                      onDragStart={(ev) => {
                        ev.dataTransfer.setData('application/geneuclidean-block', block.blockType);
                        ev.dataTransfer.effectAllowed = 'copy';
                      }}
                      title={`${block.name} — ${block.summary}\nDrag onto canvas or click + to insert`}
                      className="group rounded-xl border border-[var(--circuit-node-border)] bg-[var(--circuit-node-bg)]/90 hover:border-cyan-500/45 px-2.5 py-2 transition-all cursor-grab active:cursor-grabbing"
                    >
                      <div className="flex items-center justify-between gap-1.5">
                        <div className="min-w-0">
                          <div className="text-[11.5px] font-semibold text-[var(--text-primary)] truncate leading-tight">
                            {block.shortName}
                          </div>
                          <div className="flex items-center gap-1.5 mt-1">
                            <span
                              className="font-mono-num text-[9px] font-semibold px-1.5 py-0.2 rounded"
                              style={{ backgroundColor: `${color}18`, color }}
                            >
                              {block.badge}
                            </span>
                            {/* Miniature typed I/O pin dots */}
                            <div className="flex items-center gap-0.5">
                              {block.inputs.slice(0, 3).map((inp) => (
                                <span
                                  key={inp.id}
                                  className="h-1.5 w-1.5 rounded-full opacity-75"
                                  style={{ backgroundColor: getSignalColor(inp.signalType, theme) }}
                                />
                              ))}
                              {block.inputs.length > 0 && block.outputs.length > 0 && (
                                <span className="text-[8px] text-[var(--text-muted)] mx-0.5">→</span>
                              )}
                              {block.outputs.slice(0, 3).map((out) => (
                                <span
                                  key={out.id}
                                  className="h-1.5 w-1.5 rounded-full"
                                  style={{ backgroundColor: getSignalColor(out.signalType, theme) }}
                                />
                              ))}
                            </div>
                          </div>
                        </div>

                        <button
                          type="button"
                          data-add-block={block.blockType}
                          onClick={(ev) => {
                            ev.stopPropagation();
                            onAddBlock(block.blockType);
                          }}
                          title={`Add ${block.shortName} to canvas`}
                          className="h-6 w-6 rounded-lg border border-[var(--circuit-node-border)] bg-[var(--bg-elevated)] group-hover:border-cyan-500/50 group-hover:text-cyan-600 dark:group-hover:text-cyan-300 flex items-center justify-center text-[var(--text-muted)] transition-colors shrink-0 cursor-pointer"
                        >
                          <Plus className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {totalMatches === 0 && (
          <div className="px-3 py-8 text-center text-xs text-[var(--text-muted)]">
            No blocks match &ldquo;{query}&rdquo;.
          </div>
        )}
      </div>

      {/* Subtle Signal Legend Footer */}
      <div className="px-3 py-2 border-t border-[var(--circuit-node-border)] text-[10px] text-[var(--text-muted)] flex items-center justify-between">
        <span>Drag or click + to wire</span>
        <span className="font-mono-num">36 real APIs</span>
      </div>
    </aside>
  );
}
