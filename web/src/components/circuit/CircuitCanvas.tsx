import {
  AlertCircle,
  Copy,
  Layers,
  Map as MapIcon,
  Maximize2,
  Minus,
  Plus,
  Redo2,
  Repeat,
  Sparkles,
  Trash2,
  Undo2,
  Workflow,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ActiveWiringDraft,
  CircuitEdge,
  CircuitNode,
  ExecutionStepTelemetry,
  NodeRuntimeStatus,
  PortSpec,
  ValidationResult,
} from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { BLOCK_CATALOG_MAP, SIGNAL_TYPE_META } from '../../lib/circuitCatalog';
import {
  NODE_WIDTH,
  canConnectPorts,
  estimateNodeHeight,
  getPortAnchorPosition,
} from '../../lib/circuitEngine';
import { CircuitNodeCard } from './CircuitNodeCard';
import { CircuitWireLayer } from './CircuitWireLayer';
import { Minimap } from './Minimap';
import { getSignalColor } from './blockIcons';

interface CircuitCanvasProps {
  nodes: CircuitNode[];
  edges: CircuitEdge[];
  theme: ThemeMode;
  viewport: { x: number; y: number; zoom: number };
  onViewportChange: (vp: { x: number; y: number; zoom: number }) => void;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onSelectEdge: (edgeId: string | null) => void;
  onMoveNodeLive: (nodeId: string, position: { x: number; y: number }) => void;
  onCommitNodeMove: (nodeId: string, startPos: { x: number; y: number }, endPos: { x: number; y: number }) => void;
  onConnectEdge: (
    sourceNodeId: string,
    sourcePortId: string,
    targetNodeId: string,
    targetPortId: string,
  ) => void;
  onDropNewBlock: (blockType: string, worldPos: { x: number; y: number }) => void;
  validation: ValidationResult;
  nodeStatusMap: Record<string, NodeRuntimeStatus>;
  nodeTelemetryMap: Record<string, ExecutionStepTelemetry>;
  activeEdgeIds: Set<string>;
  isRunning: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onAutoLayout: () => void;
  onDuplicateSelected: () => void;
  onDeleteSelected: () => void;
  onQuickAddBlock: (blockType: string) => void;
  onQuickLoadPreset: (presetId: string) => void;
  fitViewSignal: number;
  focusNodeSignal: { nodeId: string; seq: number } | null;
  toastMessage: { text: string; tone: 'info' | 'feedback' | 'error' } | null;
  onDismissToast: () => void;
}

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 1.8;

export function CircuitCanvas({
  nodes,
  edges,
  theme,
  viewport,
  onViewportChange,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onMoveNodeLive,
  onCommitNodeMove,
  onConnectEdge,
  onDropNewBlock,
  validation,
  nodeStatusMap,
  nodeTelemetryMap,
  activeEdgeIds,
  isRunning,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onAutoLayout,
  onDuplicateSelected,
  onDeleteSelected,
  onQuickAddBlock,
  onQuickLoadPreset,
  fitViewSignal,
  focusNodeSignal,
  toastMessage,
  onDismissToast,
}: CircuitCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 920, height: 680 });
  const [showMinimap, setShowMinimap] = useState(true);
  const [showLegend, setShowLegend] = useState(false);
  const [spacePressed, setSpacePressed] = useState(false);

  // Panning state ref
  const panRef = useRef<{
    startClientX: number;
    startClientY: number;
    startVpX: number;
    startVpY: number;
  } | null>(null);

  // Node dragging state ref
  const dragNodeRef = useRef<{
    nodeId: string;
    startClientX: number;
    startClientY: number;
    startNodePos: { x: number; y: number };
    lastNodePos: { x: number; y: number };
    moved: boolean;
  } | null>(null);

  // Active wiring draft state
  const [wiringDraft, setWiringDraft] = useState<ActiveWiringDraft | null>(null);
  const wiringStartClientRef = useRef<{ x: number; y: number } | null>(null);
  const wiringDraftRef = useRef<ActiveWiringDraft | null>(null);
  wiringDraftRef.current = wiringDraft;

  // Track canvas container dimensions
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setCanvasSize({ width: rect.width, height: rect.height });
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Space key for Space+drag pan & Escape to cancel wiring
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT');
      if (e.code === 'Space' && !inInput) {
        setSpacePressed(true);
      } else if (e.key === 'Escape') {
        setWiringDraft(null);
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') setSpacePressed(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const screenToWorld = useCallback(
    (clientX: number, clientY: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return {
        x: (clientX - rect.left - viewport.x) / viewport.zoom,
        y: (clientY - rect.top - viewport.y) / viewport.zoom,
      };
    },
    [viewport],
  );

  const handleFitView = useCallback(() => {
    const rect = containerRef.current?.getBoundingClientRect();
    const w = rect?.width || canvasSize.width || 920;
    const h = rect?.height || canvasSize.height || 680;
    if (nodes.length === 0) {
      onViewportChange({ x: 0, y: 0, zoom: 1 });
      return;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const hasFeedback = edges.some((e) => e.kind === 'feedback');
    for (const n of nodes) {
      const nh = estimateNodeHeight(n, Boolean(nodeTelemetryMap[n.id]));
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + NODE_WIDTH);
      maxY = Math.max(maxY, n.position.y + nh);
    }
    if (hasFeedback) {
      minY -= 96; // leave breathing room for the overhead DMTA loop arc
    }

    const padX = 56;
    const padTop = 44;
    const padBottom = 86; // leave space for floating pill toolbar
    const contentW = Math.max(240, maxX - minX);
    const contentH = Math.max(180, maxY - minY);
    const scaleX = (w - padX * 2) / contentW;
    const scaleY = (h - padTop - padBottom) / contentH;
    const zoom = Math.max(MIN_ZOOM, Math.min(1.06, Math.min(scaleX, scaleY)));
    const x = (w - contentW * zoom) / 2 - minX * zoom;
    const y = padTop + (h - padTop - padBottom - contentH * zoom) / 2 - minY * zoom;
    onViewportChange({ x, y, zoom });
  }, [nodes, edges, canvasSize, nodeTelemetryMap, onViewportChange]);

  // Trigger Fit-to-View when requested by parent (preset switch, keyboard `0`, auto-layout)
  const lastFitSeqRef = useRef(0);
  useEffect(() => {
    if (fitViewSignal !== lastFitSeqRef.current) {
      lastFitSeqRef.current = fitViewSignal;
      const id = requestAnimationFrame(() => handleFitView());
      return () => cancelAnimationFrame(id);
    }
  }, [fitViewSignal, handleFitView]);

  // Pan smoothly to focused node (e.g. clicking a diagnostic or telemetry step)
  const lastFocusSeqRef = useRef(0);
  useEffect(() => {
    if (!focusNodeSignal || focusNodeSignal.seq === lastFocusSeqRef.current) return;
    lastFocusSeqRef.current = focusNodeSignal.seq;
    const target = nodes.find((n) => n.id === focusNodeSignal.nodeId);
    if (!target) return;
    const nh = estimateNodeHeight(target, Boolean(nodeTelemetryMap[target.id]));
    const cx = target.position.x + NODE_WIDTH / 2;
    const cy = target.position.y + nh / 2;
    const targetZoom = Math.max(0.85, Math.min(1.1, viewport.zoom));
    onViewportChange({
      x: canvasSize.width / 2 - cx * targetZoom,
      y: canvasSize.height / 2 - cy * targetZoom,
      zoom: targetZoom,
    });
  }, [focusNodeSignal, nodes, nodeTelemetryMap, canvasSize, viewport.zoom, onViewportChange]);

  const handleZoomBy = (factor: number, anchorClientX?: number, anchorClientY?: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const ax = anchorClientX !== undefined ? anchorClientX - rect.left : rect.width / 2;
    const ay = anchorClientY !== undefined ? anchorClientY - rect.top : rect.height / 2;
    const nextZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, viewport.zoom * factor));
    const ratio = nextZoom / viewport.zoom;
    onViewportChange({
      x: ax - (ax - viewport.x) * ratio,
      y: ay - (ay - viewport.y) * ratio,
      zoom: nextZoom,
    });
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.09 : 0.915;
    handleZoomBy(factor, e.clientX, e.clientY);
  };

  const handleCanvasPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button === 1 || e.button === 0) {
      // Cancel click-to-wire if clicking empty canvas
      if (wiringDraftRef.current) {
        setWiringDraft(null);
      }
      onSelectNode(null);
      onSelectEdge(null);
      panRef.current = {
        startClientX: e.clientX,
        startClientY: e.clientY,
        startVpX: viewport.x,
        startVpY: viewport.y,
      };
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const handleNodePointerDown = (e: React.PointerEvent, nodeId: string) => {
    if (spacePressed || e.button === 1) return; // let canvas pan
    e.stopPropagation();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;
    onSelectNode(nodeId);
    onSelectEdge(null);
    dragNodeRef.current = {
      nodeId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startNodePos: { ...node.position },
      lastNodePos: { ...node.position },
      moved: false,
    };
  };

  const checkCompatibility = useCallback(
    (targetNodeId: string, targetPortId: string) => {
      const draft = wiringDraftRef.current;
      if (!draft) return { compatible: false, wouldBeFeedback: false };
      return canConnectPorts(
        {
          schemaVersion: 'geneuclidean.circuit.v1',
          id: 'check',
          name: 'check',
          description: '',
          execution: { mode: 'browser_sim', seed: 42, dmtaRounds: 1, targetContext: 'DRD1', budget: 16 },
          nodes,
          edges,
        },
        draft.sourceNodeId,
        draft.sourcePortId,
        targetNodeId,
        targetPortId,
      );
    },
    [nodes, edges],
  );

  const handleStartWiring = (e: React.PointerEvent, nodeId: string, port: PortSpec) => {
    const srcNode = nodes.find((n) => n.id === nodeId);
    if (!srcNode) return;
    const anchor = getPortAnchorPosition(srcNode, port.id, 'output');
    wiringStartClientRef.current = { x: e.clientX, y: e.clientY };
    setWiringDraft({
      sourceNodeId: nodeId,
      sourcePortId: port.id,
      signalType: port.signalType,
      startX: anchor.x,
      startY: anchor.y,
      cursorX: anchor.x,
      cursorY: anchor.y,
      hoveredTarget: null,
    });
  };

  const handleHoverInputPort = (nodeId: string, port: PortSpec | null) => {
    setWiringDraft((prev) => {
      if (!prev) return null;
      if (!port) return { ...prev, hoveredTarget: null };
      const check = checkCompatibility(nodeId, port.id);
      return {
        ...prev,
        hoveredTarget: {
          nodeId,
          portId: port.id,
          compatible: check.compatible,
          wouldBeFeedback: check.wouldBeFeedback,
          remediation: check.remediation,
        },
      };
    });
  };

  const handleCompleteWiring = (_e: React.PointerEvent, targetNodeId: string, port: PortSpec) => {
    const draft = wiringDraftRef.current;
    if (!draft) return;
    const check = checkCompatibility(targetNodeId, port.id);
    if (check.compatible) {
      onConnectEdge(draft.sourceNodeId, draft.sourcePortId, targetNodeId, port.id);
      setWiringDraft(null);
      wiringStartClientRef.current = null;
    } else if (check.remediation) {
      // Keep hovered remediation visible briefly
      setWiringDraft((prev) =>
        prev
          ? {
              ...prev,
              hoveredTarget: {
                nodeId: targetNodeId,
                portId: port.id,
                compatible: false,
                wouldBeFeedback: false,
                remediation: check.remediation,
              },
            }
          : null,
      );
    }
  };

  // Global pointermove / pointerup for smooth dragging, panning, and wiring
  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      if (panRef.current) {
        const dx = e.clientX - panRef.current.startClientX;
        const dy = e.clientY - panRef.current.startClientY;
        onViewportChange({
          ...viewport,
          x: panRef.current.startVpX + dx,
          y: panRef.current.startVpY + dy,
        });
        return;
      }

      if (dragNodeRef.current) {
        const drag = dragNodeRef.current;
        const dx = (e.clientX - drag.startClientX) / viewport.zoom;
        const dy = (e.clientY - drag.startClientY) / viewport.zoom;
        if (Math.hypot(dx, dy) > 2) {
          drag.moved = true;
        }
        const nextPos = {
          x: Math.round(drag.startNodePos.x + dx),
          y: Math.round(drag.startNodePos.y + dy),
        };
        drag.lastNodePos = nextPos;
        onMoveNodeLive(drag.nodeId, nextPos);
        return;
      }

      if (wiringDraftRef.current) {
        const worldPos = screenToWorld(e.clientX, e.clientY);
        setWiringDraft((prev) =>
          prev ? { ...prev, cursorX: worldPos.x, cursorY: worldPos.y } : null,
        );
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      if (panRef.current) {
        panRef.current = null;
      }

      if (dragNodeRef.current) {
        const drag = dragNodeRef.current;
        dragNodeRef.current = null;
        if (drag.moved) {
          onCommitNodeMove(drag.nodeId, drag.startNodePos, drag.lastNodePos);
        }
      }

      if (wiringDraftRef.current) {
        const draft = wiringDraftRef.current;
        const startPt = wiringStartClientRef.current;
        const dragDist = startPt ? Math.hypot(e.clientX - startPt.x, e.clientY - startPt.y) : 99;
        // If user dragged > 12px and released off a compatible pin, dismiss wire
        if (dragDist > 12) {
          if (draft.hoveredTarget?.compatible) {
            onConnectEdge(
              draft.sourceNodeId,
              draft.sourcePortId,
              draft.hoveredTarget.nodeId,
              draft.hoveredTarget.portId,
            );
          }
          setWiringDraft(null);
          wiringStartClientRef.current = null;
        }
      }
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [viewport, onViewportChange, onMoveNodeLive, onCommitNodeMove, onConnectEdge, screenToWorld]);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (e.dataTransfer.types.includes('application/geneuclidean-block')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    const blockType = e.dataTransfer.getData('application/geneuclidean-block');
    if (!blockType || !BLOCK_CATALOG_MAP[blockType]) return;
    e.preventDefault();
    const world = screenToWorld(e.clientX, e.clientY);
    onDropNewBlock(blockType, {
      x: Math.round(world.x - NODE_WIDTH / 2),
      y: Math.round(world.y - 50),
    });
  };

  const errorNodeIds = new Set(
    validation.diagnostics.filter((d) => d.severity === 'error' && d.nodeId).map((d) => d.nodeId!),
  );

  // Subtle dot-matrix grid tracking viewport
  const gridStep = Math.max(12, 24 * viewport.zoom);
  const gridOffsetX = ((viewport.x % gridStep) + gridStep) % gridStep;
  const gridOffsetY = ((viewport.y % gridStep) + gridStep) % gridStep;

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onPointerDown={handleCanvasPointerDown}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{
        backgroundColor: 'var(--circuit-canvas-bg)',
        backgroundImage: `radial-gradient(circle, var(--circuit-grid-dot) 1.25px, transparent 1.25px)`,
        backgroundSize: `${gridStep}px ${gridStep}px`,
        backgroundPosition: `${gridOffsetX}px ${gridOffsetY}px`,
      }}
      className={`relative flex-1 h-full overflow-hidden select-none ${
        spacePressed ? 'cursor-grab active:cursor-grabbing' : wiringDraft ? 'cursor-crosshair' : 'cursor-default'
      }`}
    >
      {/* Subtle Ambient Corner Radiance (Obsidian Dark only) */}
      {theme === 'dark' && (
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 65% 45% at 50% 0%, rgba(56, 189, 248, 0.06), transparent 65%)',
          }}
        />
      )}

      {/* World Coordinate Transform Layer */}
      <div
        style={{
          transform: `translate3d(${viewport.x}px, ${viewport.y}px, 0) scale(${viewport.zoom})`,
          transformOrigin: '0 0',
        }}
        className="absolute inset-0 will-change-transform"
      >
        {/* SVG Wire Layer sits cleanly beneath nodes */}
        <CircuitWireLayer
          nodes={nodes}
          edges={edges}
          theme={theme}
          selectedEdgeId={selectedEdgeId}
          activeEdgeIds={activeEdgeIds}
          isRunning={isRunning}
          wiringDraft={wiringDraft}
          onSelectEdge={(edgeId) => {
            onSelectEdge(edgeId);
            onSelectNode(null);
          }}
        />

        {/* Node Cards Layer */}
        {nodes.map((node) => (
          <CircuitNodeCard
            key={node.id}
            node={node}
            theme={theme}
            selected={selectedNodeId === node.id}
            hasError={errorNodeIds.has(node.id)}
            runtimeStatus={nodeStatusMap[node.id] ?? 'idle'}
            latestTelemetry={nodeTelemetryMap[node.id]}
            wiringDraft={wiringDraft}
            isPortCompatible={checkCompatibility}
            onNodePointerDown={handleNodePointerDown}
            onStartWiring={handleStartWiring}
            onCompleteWiring={handleCompleteWiring}
            onHoverInputPort={handleHoverInputPort}
          />
        ))}
      </div>

      {/* Top-Left Compact Signal Legend Toggle */}
      <div
        className="absolute top-3 left-3 z-20 flex flex-col items-start gap-2"
        onPointerDown={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          onClick={() => setShowLegend((v) => !v)}
          className="circuit-floating-pill h-8 px-2.5 rounded-full text-[11px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] flex items-center gap-1.5 transition-colors cursor-pointer"
        >
          <Layers className="h-3.5 w-3.5 text-cyan-500" />
          <span>Signals</span>
          <span className="flex items-center -space-x-0.5 ml-0.5">
            {(['POCKET_3D', 'SE3_GRAPH', 'MOL_BATCH', 'ORACLE_SCORES', 'UNCERTAINTY_DIST'] as const).map(
              (sig) => (
                <span
                  key={sig}
                  className="h-2 w-2 rounded-full ring-1 ring-[var(--circuit-node-bg)]"
                  style={{ backgroundColor: getSignalColor(sig, theme) }}
                />
              ),
            )}
          </span>
        </button>

        {showLegend && (
          <div className="circuit-floating-pill rounded-2xl p-3 w-[240px] space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold text-[var(--text-primary)]">
                Typed Signal Grammar (10)
              </span>
              <button
                type="button"
                onClick={() => setShowLegend(false)}
                className="text-[var(--text-muted)] hover:text-[var(--text-primary)] cursor-pointer"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {Object.values(SIGNAL_TYPE_META).map((meta) => (
                <div key={meta.type} className="flex items-center gap-1.5 text-[10.5px]">
                  <span
                    className="h-2.5 w-2.5 rounded-full shrink-0"
                    style={{ backgroundColor: getSignalColor(meta.type, theme) }}
                  />
                  <span className="text-[var(--text-secondary)] truncate">{meta.label}</span>
                </div>
              ))}
            </div>
            <div className="pt-1.5 border-t border-[var(--border-subtle)] flex items-center gap-1.5 text-[10px] text-fuchsia-600 dark:text-fuchsia-300 font-medium">
              <Repeat className="h-3 w-3 shrink-0" />
              <span>Dashed arc = Legal DMTA Feedback Loop</span>
            </div>
          </div>
        )}
      </div>

      {/* Live Wiring Status / Remediation Floating Pill (Top-Center) */}
      {wiringDraft && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 pointer-events-none max-w-lg px-3">
          {wiringDraft.hoveredTarget && !wiringDraft.hoveredTarget.compatible ? (
            <div className="circuit-floating-pill rounded-2xl px-3.5 py-2 border-rose-500/45 flex items-start gap-2 text-xs">
              <AlertCircle className="h-4 w-4 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <div className="font-semibold text-rose-600 dark:text-rose-300">
                  Incompatible Signal Pin
                </div>
                <div className="text-[11px] text-[var(--text-secondary)] leading-snug mt-0.5">
                  {wiringDraft.hoveredTarget.remediation}
                </div>
              </div>
            </div>
          ) : wiringDraft.hoveredTarget?.wouldBeFeedback ? (
            <div className="circuit-floating-pill rounded-full px-3.5 py-1.5 border-fuchsia-500/50 flex items-center gap-2 text-xs font-semibold text-fuchsia-600 dark:text-fuchsia-300">
              <Repeat className="h-3.5 w-3.5" />
              Release to attach as ↻ DMTA Feedback Loop
            </div>
          ) : (
            <div className="circuit-floating-pill rounded-full px-3.5 py-1.5 flex items-center gap-2 text-[11.5px] text-[var(--text-secondary)]">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: getSignalColor(wiringDraft.signalType, theme) }}
              />
              <span>
                Wiring{' '}
                <strong className="text-[var(--text-primary)]">
                  {SIGNAL_TYPE_META[wiringDraft.signalType].label}
                </strong>{' '}
                — drop on a pulsing matching input pin (Esc to cancel)
              </span>
            </div>
          )}
        </div>
      )}

      {/* Ephemeral Toast Notification (Top-Center when not wiring) */}
      {!wiringDraft && toastMessage && (
        <div
          className="absolute top-3 left-1/2 -translate-x-1/2 z-30"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <div
            className={`circuit-floating-pill rounded-full px-3.5 py-1.5 flex items-center gap-2 text-xs font-medium ${
              toastMessage.tone === 'feedback'
                ? 'border-fuchsia-500/50 text-fuchsia-600 dark:text-fuchsia-300'
                : toastMessage.tone === 'error'
                  ? 'border-rose-500/50 text-rose-600 dark:text-rose-300'
                  : 'text-[var(--text-primary)]'
            }`}
          >
            {toastMessage.tone === 'feedback' && <Repeat className="h-3.5 w-3.5" />}
            {toastMessage.tone === 'error' && <AlertCircle className="h-3.5 w-3.5" />}
            <span>{toastMessage.text}</span>
            <button
              type="button"
              onClick={onDismissToast}
              className="opacity-60 hover:opacity-100 ml-1 cursor-pointer"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Blank Breadboard Welcome Empty State */}
      {nodes.length === 0 && (
        <div
          className="absolute inset-0 flex items-center justify-center p-6 pointer-events-none"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <div className="circuit-floating-pill pointer-events-auto rounded-3xl p-6 max-w-md w-full text-center space-y-4">
            <div className="mx-auto h-11 w-11 rounded-2xl bg-gradient-to-br from-sky-500/20 to-emerald-500/20 border border-cyan-500/30 flex items-center justify-center">
              <Workflow className="h-5 w-5 text-cyan-500" />
            </div>
            <div className="space-y-1">
              <h2 className="text-base font-bold tracking-tight text-[var(--text-primary)]">
                Blank GenEuclidean Breadboard
              </h2>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Drag any of the 36 real GenEuclidean blocks from the left dock, connect matching
                signal pins, and live-generate runnable Python SDK pipelines.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <button
                type="button"
                onClick={() => onQuickAddBlock('protein_pocket')}
                className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-500/35 hover:bg-cyan-500/25 transition-colors cursor-pointer"
              >
                + Add Protein Pocket
              </button>
              <button
                type="button"
                onClick={() => onQuickLoadPreset('se3_denovo_cns')}
                className="px-3 py-1.5 rounded-xl text-xs font-semibold lab-card text-[var(--text-primary)] hover:border-cyan-500/45 flex items-center gap-1.5 cursor-pointer"
              >
                <Sparkles className="h-3.5 w-3.5 text-emerald-500" />
                Load SE(3) De Novo
              </button>
              <button
                type="button"
                onClick={() => onQuickLoadPreset('al_dmta_closed_loop')}
                className="px-3 py-1.5 rounded-xl text-xs font-semibold lab-card text-[var(--text-primary)] hover:border-fuchsia-500/45 flex items-center gap-1.5 cursor-pointer"
              >
                <Repeat className="h-3.5 w-3.5 text-fuchsia-500" />
                Load DMTA Loop
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Floating Bottom-Left Interactive Minimap */}
      {showMinimap && nodes.length > 0 && (
        <div
          className="absolute bottom-4 left-4 z-20 hidden md:block"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <Minimap
            nodes={nodes}
            edges={edges}
            viewport={viewport}
            canvasSize={canvasSize}
            theme={theme}
            selectedNodeId={selectedNodeId}
            onJumpToWorld={(wx, wy) => {
              onViewportChange({
                ...viewport,
                x: canvasSize.width / 2 - wx * viewport.zoom,
                y: canvasSize.height / 2 - wy * viewport.zoom,
              });
            }}
          />
        </div>
      )}

      {/* Floating Bottom-Center Apple Pill Toolbar */}
      <div
        className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20"
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="circuit-floating-pill rounded-full px-2 py-1.5 flex items-center gap-0.5 text-[var(--text-secondary)]">
          <button
            type="button"
            onClick={() => handleZoomBy(0.85)}
            title="Zoom Out"
            aria-label="Zoom Out"
            className="h-7 w-7 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] flex items-center justify-center transition-colors cursor-pointer"
          >
            <Minus className="h-3.5 w-3.5" />
          </button>
          <span className="font-mono-num text-[11px] font-semibold text-[var(--text-primary)] w-11 text-center">
            {Math.round(viewport.zoom * 100)}%
          </span>
          <button
            type="button"
            onClick={() => handleZoomBy(1.18)}
            title="Zoom In"
            aria-label="Zoom In"
            className="h-7 w-7 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] flex items-center justify-center transition-colors cursor-pointer"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>

          <div className="w-px h-4 bg-[var(--border-subtle)] mx-1" />

          <button
            type="button"
            onClick={handleFitView}
            title="Fit to View (0)"
            className="h-7 px-2 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] flex items-center gap-1 text-[11px] font-medium transition-colors cursor-pointer"
          >
            <Maximize2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Fit</span>
          </button>
          <button
            type="button"
            onClick={onAutoLayout}
            title="Topological Auto-Layout (L)"
            className="h-7 px-2 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] flex items-center gap-1 text-[11px] font-medium transition-colors cursor-pointer"
          >
            <Workflow className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Auto-Layout</span>
          </button>

          <div className="w-px h-4 bg-[var(--border-subtle)] mx-1" />

          <button
            type="button"
            disabled={!canUndo}
            onClick={onUndo}
            title="Undo (⌘Z)"
            aria-label="Undo"
            className="h-7 w-7 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] disabled:opacity-35 disabled:cursor-not-allowed flex items-center justify-center transition-colors cursor-pointer"
          >
            <Undo2 className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            disabled={!canRedo}
            onClick={onRedo}
            title="Redo (⇧⌘Z)"
            aria-label="Redo"
            className="h-7 w-7 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] disabled:opacity-35 disabled:cursor-not-allowed flex items-center justify-center transition-colors cursor-pointer"
          >
            <Redo2 className="h-3.5 w-3.5" />
          </button>

          <div className="w-px h-4 bg-[var(--border-subtle)] mx-1" />

          <button
            type="button"
            disabled={!selectedNodeId}
            onClick={onDuplicateSelected}
            title="Duplicate Block (⌘D)"
            aria-label="Duplicate Selected Block"
            className="h-7 w-7 rounded-full hover:bg-slate-500/12 hover:text-[var(--text-primary)] disabled:opacity-35 disabled:cursor-not-allowed flex items-center justify-center transition-colors cursor-pointer"
          >
            <Copy className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            disabled={!selectedNodeId && !selectedEdgeId}
            onClick={onDeleteSelected}
            title="Delete Selected (⌫)"
            aria-label="Delete Selected"
            className="h-7 w-7 rounded-full hover:bg-rose-500/15 hover:text-rose-500 disabled:opacity-35 disabled:cursor-not-allowed flex items-center justify-center transition-colors cursor-pointer"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>

          <div className="w-px h-4 bg-[var(--border-subtle)] mx-1 hidden md:block" />

          <button
            type="button"
            onClick={() => setShowMinimap((v) => !v)}
            title="Toggle Minimap"
            aria-label="Toggle Minimap"
            className={`h-7 w-7 rounded-full hidden md:flex items-center justify-center transition-colors cursor-pointer ${
              showMinimap
                ? 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300'
                : 'hover:bg-slate-500/12 hover:text-[var(--text-primary)]'
            }`}
          >
            <MapIcon className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
