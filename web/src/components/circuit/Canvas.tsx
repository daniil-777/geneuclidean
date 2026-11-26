import {
  AlertCircle,
  Copy,
  Layers,
  Map as MapIcon,
  Maximize2,
  Minus,
  Plus,
  Redo2,
  Sparkles,
  Trash2,
  Undo2,
  Wand2,
} from 'lucide-react';
import type React from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  CircuitGraphV1,
  ExecutionStepTelemetry,
  NodeRuntimeStatus,
  PortSpec,
  ValidationResult,
  WiringDraft,
} from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { BLOCK_MAP } from '../../lib/circuitCatalog';
import {
  evaluatePortConnection,
  getPortPosition,
  NODE_WIDTH,
} from '../../lib/circuitEngine';
import { CircuitNode } from './CircuitNode';
import { CircuitWire } from './CircuitWire';
import { Minimap } from './Minimap';

interface CanvasProps {
  circuit: CircuitGraphV1;
  validation: ValidationResult;
  theme: ThemeMode;
  viewport: { x: number; y: number; zoom: number };
  onChangeViewport: (vp: { x: number; y: number; zoom: number }) => void;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onSelectEdge: (edgeId: string | null) => void;
  onMoveNodePreview: (nodeId: string, pos: { x: number; y: number }) => void;
  onCommitNodeMove: (nodeId: string, startPos: { x: number; y: number }, endPos: { x: number; y: number }) => void;
  onConnectPorts: (
    sourceNodeId: string,
    sourcePortId: string,
    targetNodeId: string,
    targetPortId: string,
  ) => void;
  onWiringRejected: (reason: string) => void;
  onDropNewBlock: (blockType: string, worldPos: { x: number; y: number }) => void;
  onFitView: () => void;
  onAutoLayout: () => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onDuplicateSelected: () => void;
  onDeleteSelected: () => void;
  onLoadDefaultPreset: () => void;
  nodeStatuses: Record<string, NodeRuntimeStatus>;
  nodeTelemetry: Record<string, ExecutionStepTelemetry>;
  activeEdgeIds: Set<string>;
  isRunning: boolean;
  toastMessage: { text: string; tone: 'info' | 'error' } | null;
  canvasContainerRef: React.RefObject<HTMLDivElement | null>;
}

const MIN_ZOOM = 0.4;
const MAX_ZOOM = 1.8;

export function Canvas({
  circuit,
  validation,
  theme,
  viewport,
  onChangeViewport,
  selectedNodeId,
  selectedEdgeId,
  onSelectNode,
  onSelectEdge,
  onMoveNodePreview,
  onCommitNodeMove,
  onConnectPorts,
  onWiringRejected,
  onDropNewBlock,
  onFitView,
  onAutoLayout,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  onDuplicateSelected,
  onDeleteSelected,
  onLoadDefaultPreset,
  nodeStatuses,
  nodeTelemetry,
  activeEdgeIds,
  isRunning,
  toastMessage,
  canvasContainerRef,
}: CanvasProps) {
  const [showMinimap, setShowMinimap] = useState(true);
  const [spacePressed, setSpacePressed] = useState(false);
  const [wiringDraft, setWiringDraft] = useState<WiringDraft | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 960, height: 680 });

  const panStateRef = useRef<{
    active: boolean;
    startClientX: number;
    startClientY: number;
    startVpX: number;
    startVpY: number;
  } | null>(null);

  const dragNodeRef = useRef<{
    nodeId: string;
    startClientX: number;
    startClientY: number;
    origX: number;
    origY: number;
    lastX: number;
    lastY: number;
    moved: boolean;
  } | null>(null);

  const wiringOriginTimeRef = useRef<number>(0);

  // Track container dimensions for Minimap & Fit-to-View
  useEffect(() => {
    const el = canvasContainerRef.current;
    if (!el) return;
    const update = () => setCanvasSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [canvasContainerRef]);

  // Track Spacebar for hand-panning and Escape to cancel active wire
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement)) {
        setSpacePressed(true);
      }
      if (e.key === 'Escape') {
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

  const clientToWorld = useCallback(
    (clientX: number, clientY: number) => {
      const rect = canvasContainerRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return {
        x: (clientX - rect.left - viewport.x) / viewport.zoom,
        y: (clientY - rect.top - viewport.y) / viewport.zoom,
      };
    },
    [canvasContainerRef, viewport],
  );

  // Cursor-centered smooth wheel zoom
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const rect = canvasContainerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = e.deltaY < 0 ? 1.09 : 0.917;
    const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((viewport.zoom * delta).toFixed(3))));
    const scaleRatio = nextZoom / viewport.zoom;
    onChangeViewport({
      x: mx - (mx - viewport.x) * scaleRatio,
      y: my - (my - viewport.y) * scaleRatio,
      zoom: nextZoom,
    });
  };

  const handleCanvasPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Clicking empty breadboard clears active wiring & selection, and starts pan
    if (wiringDraft) {
      setWiringDraft(null);
      return;
    }
    if (e.button === 0 || e.button === 1 || spacePressed) {
      onSelectNode(null);
      onSelectEdge(null);
      panStateRef.current = {
        active: true,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startVpX: viewport.x,
        startVpY: viewport.y,
      };
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    }
  };

  const handleNodePointerDown = (e: React.PointerEvent, nodeId: string) => {
    if (spacePressed || e.button === 1) return;
    e.stopPropagation();
    const node = circuit.nodes.find((n) => n.id === nodeId);
    if (!node) return;
    onSelectEdge(null);
    onSelectNode(nodeId);
    dragNodeRef.current = {
      nodeId,
      startClientX: e.clientX,
      startClientY: e.clientY,
      origX: node.position.x,
      origY: node.position.y,
      lastX: node.position.x,
      lastY: node.position.y,
      moved: false,
    };
  };

  const handleStartWiring = (e: React.PointerEvent, nodeId: string, port: PortSpec) => {
    const node = circuit.nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const anchor = getPortPosition(node, port.id, 'output');
    wiringOriginTimeRef.current = Date.now();
    setWiringDraft({
      sourceNodeId: nodeId,
      sourcePortId: port.id,
      signalType: port.signalType,
      startX: anchor.x,
      startY: anchor.y,
      cursorX: anchor.x + 12,
      cursorY: anchor.y,
      hoveredTarget: null,
    });
    // Ensure selection focuses source node
    onSelectNode(nodeId);
    e.stopPropagation();
  };

  const findNearestInputPort = useCallback(
    (wx: number, wy: number, excludeNodeId: string) => {
      let best: { nodeId: string; portId: string; x: number; y: number; dist: number } | null = null;
      const SNAP_RADIUS = 30;
      for (const n of circuit.nodes) {
        if (n.id === excludeNodeId) continue;
        const def = BLOCK_MAP[n.blockType];
        if (!def) continue;
        for (const inp of def.inputs) {
          const pos = getPortPosition(n, inp.id, 'input');
          const d = Math.hypot(wx - pos.x, wy - pos.y);
          if (d <= SNAP_RADIUS && (!best || d < best.dist)) {
            best = { nodeId: n.id, portId: inp.id, x: pos.x, y: pos.y, dist: d };
          }
        }
      }
      return best;
    },
    [circuit.nodes],
  );

  const handleCompleteWiring = (e: React.PointerEvent, targetNodeId: string, targetPort: PortSpec) => {
    e.stopPropagation();
    if (!wiringDraft) return;
    const check = evaluatePortConnection(
      circuit,
      wiringDraft.sourceNodeId,
      wiringDraft.sourcePortId,
      targetNodeId,
      targetPort.id,
    );
    if (check.compatible) {
      onConnectPorts(wiringDraft.sourceNodeId, wiringDraft.sourcePortId, targetNodeId, targetPort.id);
    } else if (check.hint) {
      onWiringRejected(check.hint);
    }
    setWiringDraft(null);
  };

  const handleHoverInputPort = (targetNodeId: string, port: PortSpec | null) => {
    if (!wiringDraft) return;
    if (!port) {
      setWiringDraft((prev) => (prev ? { ...prev, hoveredTarget: null } : null));
      return;
    }
    const check = evaluatePortConnection(
      circuit,
      wiringDraft.sourceNodeId,
      wiringDraft.sourcePortId,
      targetNodeId,
      port.id,
    );
    const targetNode = circuit.nodes.find((n) => n.id === targetNodeId);
    const snap = targetNode ? getPortPosition(targetNode, port.id, 'input') : null;
    setWiringDraft((prev) =>
      prev
        ? {
            ...prev,
            cursorX: snap ? snap.x : prev.cursorX,
            cursorY: snap ? snap.y : prev.cursorY,
            hoveredTarget: {
              nodeId: targetNodeId,
              portId: port.id,
              compatible: check.compatible,
              willBeFeedback: check.kind === 'feedback',
              hint: check.hint,
            },
          }
        : null,
    );
  };

  const handleCanvasPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (panStateRef.current?.active) {
      const dx = e.clientX - panStateRef.current.startClientX;
      const dy = e.clientY - panStateRef.current.startClientY;
      onChangeViewport({
        ...viewport,
        x: panStateRef.current.startVpX + dx,
        y: panStateRef.current.startVpY + dy,
      });
      return;
    }

    if (dragNodeRef.current) {
      const dx = (e.clientX - dragNodeRef.current.startClientX) / viewport.zoom;
      const dy = (e.clientY - dragNodeRef.current.startClientY) / viewport.zoom;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
        dragNodeRef.current.moved = true;
      }
      const nextX = Math.round(dragNodeRef.current.origX + dx);
      const nextY = Math.round(dragNodeRef.current.origY + dy);
      dragNodeRef.current.lastX = nextX;
      dragNodeRef.current.lastY = nextY;
      onMoveNodePreview(dragNodeRef.current.nodeId, { x: nextX, y: nextY });
      return;
    }

    if (wiringDraft) {
      const pt = clientToWorld(e.clientX, e.clientY);
      const nearest = findNearestInputPort(pt.x, pt.y, wiringDraft.sourceNodeId);
      if (nearest) {
        const check = evaluatePortConnection(
          circuit,
          wiringDraft.sourceNodeId,
          wiringDraft.sourcePortId,
          nearest.nodeId,
          nearest.portId,
        );
        setWiringDraft((prev) =>
          prev
            ? {
                ...prev,
                cursorX: nearest.x,
                cursorY: nearest.y,
                hoveredTarget: {
                  nodeId: nearest.nodeId,
                  portId: nearest.portId,
                  compatible: check.compatible,
                  willBeFeedback: check.kind === 'feedback',
                  hint: check.hint,
                },
              }
            : null,
        );
      } else {
        setWiringDraft((prev) =>
          prev ? { ...prev, cursorX: pt.x, cursorY: pt.y, hoveredTarget: null } : null,
        );
      }
    }
  };

  const handleCanvasPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (panStateRef.current?.active) {
      panStateRef.current = null;
    }
    if (dragNodeRef.current) {
      const d = dragNodeRef.current;
      dragNodeRef.current = null;
      if (d.moved) {
        onCommitNodeMove(d.nodeId, { x: d.origX, y: d.origY }, { x: d.lastX, y: d.lastY });
      }
    }
    if (wiringDraft) {
      const pt = clientToWorld(e.clientX, e.clientY);
      const nearest = findNearestInputPort(pt.x, pt.y, wiringDraft.sourceNodeId);
      const targetRef = nearest
        ? { nodeId: nearest.nodeId, portId: nearest.portId }
        : wiringDraft.hoveredTarget
          ? { nodeId: wiringDraft.hoveredTarget.nodeId, portId: wiringDraft.hoveredTarget.portId }
          : null;

      if (targetRef) {
        const check = evaluatePortConnection(
          circuit,
          wiringDraft.sourceNodeId,
          wiringDraft.sourcePortId,
          targetRef.nodeId,
          targetRef.portId,
        );
        if (check.compatible) {
          onConnectPorts(
            wiringDraft.sourceNodeId,
            wiringDraft.sourcePortId,
            targetRef.nodeId,
            targetRef.portId,
          );
        } else if (check.hint) {
          onWiringRejected(check.hint);
        }
        setWiringDraft(null);
      } else if (Date.now() - wiringOriginTimeRef.current > 240) {
        setWiringDraft(null);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    if (
      e.dataTransfer.types.includes('application/x-geneuclidean-block') ||
      e.dataTransfer.types.includes('text/plain')
    ) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    let blockType = e.dataTransfer.getData('application/x-geneuclidean-block');
    if (!blockType) {
      const txt = e.dataTransfer.getData('text/plain');
      if (txt.startsWith('geneuclidean-block:')) {
        blockType = txt.slice('geneuclidean-block:'.length);
      }
    }
    if (!blockType) return;
    const pt = clientToWorld(e.clientX, e.clientY);
    onDropNewBlock(blockType, { x: pt.x - NODE_WIDTH / 2, y: pt.y - 40 });
  };

  const errorNodeIds = new Set(
    validation.diagnostics
      .filter((d) => d.severity === 'error' && d.nodeId)
      .map((d) => d.nodeId!),
  );

  const checkInputCompat = (targetNodeId: string, targetPortId: string) => {
    if (!wiringDraft) return { compatible: false, kind: 'forward' as const };
    return evaluatePortConnection(
      circuit,
      wiringDraft.sourceNodeId,
      wiringDraft.sourcePortId,
      targetNodeId,
      targetPortId,
    );
  };

  const zoomStep = (delta: number) => {
    const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number((viewport.zoom + delta).toFixed(2))));
    const cx = canvasSize.width / 2;
    const cy = canvasSize.height / 2;
    const r = nextZoom / viewport.zoom;
    onChangeViewport({
      x: cx - (cx - viewport.x) * r,
      y: cy - (cy - viewport.y) * r,
      zoom: nextZoom,
    });
  };

  const gridStep = Math.max(12, 24 * viewport.zoom);
  const hintScreenX = wiringDraft ? wiringDraft.cursorX * viewport.zoom + viewport.x : 0;
  const hintScreenY = wiringDraft ? wiringDraft.cursorY * viewport.zoom + viewport.y : 0;

  return (
    <div
      ref={canvasContainerRef}
      data-circuit-canvas="true"
      onWheel={handleWheel}
      onPointerDown={handleCanvasPointerDown}
      onPointerMove={handleCanvasPointerMove}
      onPointerUp={handleCanvasPointerUp}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{
        backgroundColor: 'var(--circuit-canvas-bg)',
        backgroundImage: 'radial-gradient(circle, var(--circuit-grid-dot) 1.15px, transparent 1.15px)',
        backgroundSize: `${gridStep}px ${gridStep}px`,
        backgroundPosition: `${viewport.x}px ${viewport.y}px`,
      }}
      className={`relative flex-1 h-full overflow-hidden select-none ${
        spacePressed ? 'cursor-grab active:cursor-grabbing' : 'cursor-default'
      }`}
    >
      {/* World Transform Layer (Wires + Nodes) */}
      <div
        style={{
          transform: `translate3d(${viewport.x}px, ${viewport.y}px, 0) scale(${viewport.zoom})`,
          transformOrigin: '0 0',
        }}
        className="absolute top-0 left-0 will-change-transform"
      >
        <CircuitWire
          nodes={circuit.nodes}
          edges={circuit.edges}
          theme={theme}
          selectedEdgeId={selectedEdgeId}
          activeEdgeIds={activeEdgeIds}
          isRunning={isRunning}
          wiringDraft={wiringDraft}
          onSelectEdge={(edgeId) => {
            onSelectNode(null);
            onSelectEdge(edgeId);
          }}
        />

        {circuit.nodes.map((node) => (
          <CircuitNode
            key={node.id}
            node={node}
            theme={theme}
            selected={selectedNodeId === node.id}
            hasError={errorNodeIds.has(node.id)}
            runtimeStatus={nodeStatuses[node.id] ?? 'idle'}
            latestTelemetry={nodeTelemetry[node.id]}
            wiringDraft={wiringDraft}
            checkInputCompat={checkInputCompat}
            onNodePointerDown={handleNodePointerDown}
            onStartWiring={handleStartWiring}
            onCompleteWiring={handleCompleteWiring}
            onHoverInputPort={handleHoverInputPort}
          />
        ))}
      </div>

      {/* Live Apple Remediation / Auto-Feedback Hint Pill Positioned Above Cursor */}
      {wiringDraft?.hoveredTarget?.hint && (
        <div
          data-wiring-hint="true"
          style={{
            left: Math.min(canvasSize.width - 286, Math.max(16, hintScreenX - 40)),
            top: Math.max(46, Math.min(canvasSize.height - 96, hintScreenY - 76)),
          }}
          className={`absolute z-40 max-w-[274px] px-3 py-2 rounded-xl text-[11px] leading-snug pointer-events-none circuit-floating-pill border ${
            !wiringDraft.hoveredTarget.compatible
              ? 'border-rose-500/55 text-rose-700 dark:text-rose-200'
              : wiringDraft.hoveredTarget.willBeFeedback
                ? 'border-fuchsia-500/55 text-fuchsia-700 dark:text-fuchsia-200'
                : 'border-emerald-500/50 text-emerald-700 dark:text-emerald-200'
          }`}
        >
          <div className="flex items-center gap-1.5 font-bold text-[10.5px] uppercase tracking-wide mb-0.5">
            {!wiringDraft.hoveredTarget.compatible ? (
              <AlertCircle className="h-3 w-3 text-rose-500 shrink-0" />
            ) : (
              <Sparkles className="h-3 w-3 shrink-0" />
            )}
            <span>
              {!wiringDraft.hoveredTarget.compatible
                ? 'Incompatible Signal'
                : wiringDraft.hoveredTarget.willBeFeedback
                  ? 'DMTA Feedback Loop'
                  : 'Compatible Socket'}
            </span>
          </div>
          <div className="font-medium opacity-95">{wiringDraft.hoveredTarget.hint}</div>
        </div>
      )}

      {/* Active Wiring Status Banner */}
      {wiringDraft && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 px-3 py-1 rounded-full circuit-floating-pill text-[11px] font-medium text-[var(--text-primary)] flex items-center gap-2 pointer-events-none">
          <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
          <span>
            Wiring <strong className="font-mono-num">{wiringDraft.signalType}</strong> — drop or click a
            pulsing compatible input socket (Esc to cancel)
          </span>
        </div>
      )}

      {/* Subtle Floating Toast Pill (DMTA Feedback, Remediation Alert, or Run Status) */}
      {toastMessage && !wiringDraft && (
        <div
          data-circuit-toast="true"
          data-toast-tone={toastMessage.tone}
          className={`absolute top-3 left-1/2 -translate-x-1/2 z-30 max-w-[460px] px-3.5 py-1.5 rounded-full circuit-floating-pill text-[11.5px] font-semibold flex items-center gap-2 pointer-events-none border ${
            toastMessage.tone === 'error'
              ? 'border-rose-500/45 text-rose-700 dark:text-rose-200'
              : 'border-[var(--circuit-node-border)] text-cyan-700 dark:text-cyan-200'
          }`}
        >
          {toastMessage.tone === 'error' ? (
            <AlertCircle className="h-3.5 w-3.5 text-rose-500 shrink-0" />
          ) : (
            <Sparkles className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
          )}
          <span className="truncate">{toastMessage.text}</span>
        </div>
      )}

      {/* Blank Breadboard Apple Frosted Welcome Card */}
      {circuit.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center p-6 pointer-events-none">
          <div className="pointer-events-auto max-w-md w-full rounded-2xl circuit-floating-pill p-6 text-center space-y-4">
            <div className="mx-auto h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-emerald-500/20 border border-cyan-500/35 flex items-center justify-center text-cyan-500">
              <Layers className="h-6 w-6" />
            </div>
            <div className="space-y-1.5">
              <h3 className="text-base font-bold tracking-tight text-[var(--text-primary)]">
                Blank GenEuclidean Breadboard
              </h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">
                Drag any of the <strong>36 real GenEuclidean blocks</strong> from the left dock
                (or click <code className="font-mono-num">+</code>) to wire custom SE(3), Active
                Learning DMTA, 7-Axis Pose Audit, and OpenFE circuits.
              </p>
            </div>
            <button
              type="button"
              onClick={onLoadDefaultPreset}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-950 shadow-md shadow-cyan-500/20 hover:brightness-110 cursor-pointer"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Load SE(3) De Novo Flagship Circuit
            </button>
          </div>
        </div>
      )}

      {/* Floating Bottom-Left Interactive Minimap */}
      {showMinimap && (
        <Minimap
          nodes={circuit.nodes}
          edges={circuit.edges}
          viewport={viewport}
          canvasSize={canvasSize}
          selectedNodeId={selectedNodeId}
          theme={theme}
          onJumpToWorldPoint={(wx, wy) =>
            onChangeViewport({
              ...viewport,
              x: canvasSize.width / 2 - wx * viewport.zoom,
              y: canvasSize.height / 2 - wy * viewport.zoom,
            })
          }
        />
      )}

      {/* Floating Bottom-Center Apple Pill Toolbar */}
      <div
        data-canvas-toolbar="true"
        onPointerDown={(e) => e.stopPropagation()}
        className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 h-10 px-1.5 rounded-2xl circuit-floating-pill flex items-center gap-0.5"
      >
        <button
          type="button"
          onClick={() => zoomStep(-0.15)}
          title="Zoom Out"
          aria-label="Zoom Out"
          className="h-7 w-7 rounded-xl hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <Minus className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={onFitView}
          title="Fit Circuit to View (0)"
          className="h-7 px-2 rounded-xl hover:bg-slate-500/12 font-mono-num text-[11px] font-semibold text-[var(--text-primary)] cursor-pointer"
        >
          {Math.round(viewport.zoom * 100)}%
        </button>
        <button
          type="button"
          onClick={() => zoomStep(0.15)}
          title="Zoom In"
          aria-label="Zoom In"
          className="h-7 w-7 rounded-xl hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>

        <div className="w-[1px] h-4 bg-[var(--circuit-node-border)] mx-1" />

        <button
          type="button"
          onClick={onFitView}
          title="Fit to View (0)"
          aria-label="Fit to View"
          className="h-7 px-2 rounded-xl hover:bg-slate-500/12 flex items-center gap-1 text-[11px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <Maximize2 className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Fit</span>
        </button>

        <button
          type="button"
          data-auto-layout="true"
          onClick={onAutoLayout}
          title="Auto-Layout Topological Layers (L)"
          aria-label="Auto-Layout"
          className="h-7 px-2 rounded-xl hover:bg-slate-500/12 flex items-center gap-1 text-[11px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
        >
          <Wand2 className="h-3.5 w-3.5 text-cyan-500" />
          <span className="hidden sm:inline">Layout</span>
        </button>

        <div className="w-[1px] h-4 bg-[var(--circuit-node-border)] mx-1" />

        <button
          type="button"
          disabled={!canUndo}
          onClick={onUndo}
          title="Undo (⌘Z)"
          aria-label="Undo"
          className="h-7 w-7 rounded-xl hover:bg-slate-500/12 disabled:opacity-35 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer disabled:cursor-not-allowed"
        >
          <Undo2 className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          disabled={!canRedo}
          onClick={onRedo}
          title="Redo (⇧⌘Z)"
          aria-label="Redo"
          className="h-7 w-7 rounded-xl hover:bg-slate-500/12 disabled:opacity-35 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer disabled:cursor-not-allowed"
        >
          <Redo2 className="h-3.5 w-3.5" />
        </button>

        <div className="w-[1px] h-4 bg-[var(--circuit-node-border)] mx-1" />

        <button
          type="button"
          disabled={!selectedNodeId}
          onClick={onDuplicateSelected}
          title="Duplicate Selected Block (⌘D)"
          aria-label="Duplicate"
          className="h-7 w-7 rounded-xl hover:bg-slate-500/12 disabled:opacity-35 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer disabled:cursor-not-allowed"
        >
          <Copy className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          disabled={!selectedNodeId && !selectedEdgeId}
          onClick={onDeleteSelected}
          title="Delete Selected (⌫)"
          aria-label="Delete"
          className="h-7 w-7 rounded-xl hover:bg-rose-500/15 disabled:opacity-35 flex items-center justify-center text-[var(--text-secondary)] hover:text-rose-500 cursor-pointer disabled:cursor-not-allowed"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>

        <div className="w-[1px] h-4 bg-[var(--circuit-node-border)] mx-1" />

        <button
          type="button"
          onClick={() => setShowMinimap((s) => !s)}
          title="Toggle Minimap"
          aria-label="Toggle Minimap"
          className={`h-7 w-7 rounded-xl flex items-center justify-center cursor-pointer transition-colors ${
            showMinimap
              ? 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300'
              : 'hover:bg-slate-500/12 text-[var(--text-muted)]'
          }`}
        >
          <MapIcon className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
