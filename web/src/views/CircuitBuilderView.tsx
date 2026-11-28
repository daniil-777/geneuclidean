import { Activity, Code2, PanelRightClose, PanelRightOpen, Sliders } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BlockPalette } from '../components/circuit/BlockPalette';
import { Canvas } from '../components/circuit/Canvas';
import { CodeExportDrawer } from '../components/circuit/CodeExportDrawer';
import { ExecutionTelemetry } from '../components/circuit/ExecutionTelemetry';
import { InspectorPanel } from '../components/circuit/InspectorPanel';
import { PresetBar } from '../components/circuit/PresetBar';
import {
  BLOCK_MAP,
  CIRCUIT_PRESETS,
  cloneCircuitGraph,
  createNodeFromBlock,
} from '../lib/circuitCatalog';
import {
  autoLayoutCircuit,
  checkLocalEngineHealth,
  evaluatePortConnection,
  exportCircuitJson,
  generatePythonSdk,
  getNodeHeight,
  NODE_WIDTH,
  parseCircuitJson,
  runCircuitOnLocalEngine,
  simulateCircuitInBrowser,
  validateCircuit,
} from '../lib/circuitEngine';
import { downloadTextFile } from '../lib/data';
import type {
  CircuitEdge,
  CircuitGraphV1,
  ExecutionEngineMode,
  ExecutionStepTelemetry,
  LocalEngineHealth,
  NodeRuntimeStatus,
  RightDrawerTab,
  RunCircuitResponse,
} from '../types/circuit';
import type { MoleculeIndexEntry, ThemeMode, V4BenchmarksData } from '../types/studio';

interface CircuitBuilderViewProps {
  molecules: MoleculeIndexEntry[];
  v4Benchmarks: V4BenchmarksData | null;
  theme: ThemeMode;
  onInspectIn3D: (target: string) => void;
}

const MAX_HISTORY = 50;

export function CircuitBuilderView({
  molecules,
  v4Benchmarks,
  theme,
  onInspectIn3D,
}: CircuitBuilderViewProps) {
  const [circuit, setCircuit] = useState<CircuitGraphV1>(() =>
    cloneCircuitGraph(CIRCUIT_PRESETS[0].graph),
  );
  const [activePresetId, setActivePresetId] = useState<string | null>(CIRCUIT_PRESETS[0].id);
  const [past, setPast] = useState<CircuitGraphV1[]>([]);
  const [future, setFuture] = useState<CircuitGraphV1[]>([]);

  const [viewport, setViewport] = useState({ x: 32, y: 40, zoom: 0.78 });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(
    CIRCUIT_PRESETS[0].graph.nodes[0]?.id ?? null,
  );
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [drawerCollapsed, setDrawerCollapsed] = useState(false);
  const [rightTab, setRightTab] = useState<RightDrawerTab>('inspector');

  const [localHealth, setLocalHealth] = useState<LocalEngineHealth>({
    reachable: false,
    checking: false,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [runResult, setRunResult] = useState<RunCircuitResponse | null>(null);
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, NodeRuntimeStatus>>({});
  const [nodeTelemetry, setNodeTelemetry] = useState<Record<string, ExecutionStepTelemetry>>({});
  const [activeEdgeIds, setActiveEdgeIds] = useState<Set<string>>(new Set());
  const [toastMessage, setToastMessage] = useState<{ text: string; tone: 'info' | 'error' } | null>(
    null,
  );

  const canvasContainerRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const showToast = useCallback((text: string, tone: 'info' | 'error' = 'info') => {
    setToastMessage({ text, tone });
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToastMessage(null), tone === 'error' ? 3800 : 2600);
  }, []);

  // Push undoable state transition
  const commitGraph = useCallback((next: CircuitGraphV1) => {
    setCircuit((prev) => {
      setPast((p) => [...p.slice(-(MAX_HISTORY - 1)), prev]);
      setFuture([]);
      return next;
    });
  }, []);

  const handleUndo = useCallback(() => {
    setPast((p) => {
      if (p.length === 0) return p;
      const previous = p[p.length - 1];
      setFuture((f) => [circuit, ...f.slice(0, MAX_HISTORY - 1)]);
      setCircuit(previous);
      return p.slice(0, -1);
    });
  }, [circuit]);

  const handleRedo = useCallback(() => {
    setFuture((f) => {
      if (f.length === 0) return f;
      const next = f[0];
      setPast((p) => [...p.slice(-(MAX_HISTORY - 1)), circuit]);
      setCircuit(next);
      return f.slice(1);
    });
  }, [circuit]);

  const validation = useMemo(() => validateCircuit(circuit), [circuit]);
  const codegen = useMemo(() => generatePythonSdk(circuit), [circuit]);

  const selectedNode = useMemo(
    () => circuit.nodes.find((n) => n.id === selectedNodeId) ?? null,
    [circuit.nodes, selectedNodeId],
  );
  const selectedEdge = useMemo(
    () => circuit.edges.find((e) => e.id === selectedEdgeId) ?? null,
    [circuit.edges, selectedEdgeId],
  );

  // Fit circuit cleanly into visible canvas bounds
  const fitGraphToCanvas = useCallback((graph: CircuitGraphV1) => {
    const el = canvasContainerRef.current;
    if (!el || graph.nodes.length === 0) {
      setViewport({ x: 32, y: 40, zoom: 0.9 });
      return;
    }
    const cw = Math.max(400, el.clientWidth);
    const ch = Math.max(300, el.clientHeight);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const hasFb = graph.edges.some((e) => e.kind === 'feedback');
    for (const n of graph.nodes) {
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y - (hasFb ? 88 : 0));
      maxX = Math.max(maxX, n.position.x + NODE_WIDTH);
      maxY = Math.max(maxY, n.position.y + getNodeHeight(n, false));
    }
    const padX = 30;
    const padTop = 28;
    const padBottom = 74;
    const bw = Math.max(240, maxX - minX);
    const bh = Math.max(180, maxY - minY);
    const zoom = Math.min(
      1.05,
      Math.max(0.35, Math.min((cw - padX * 2) / bw, (ch - padTop - padBottom) / bh)),
    );
    const x = Math.round((cw - bw * zoom) / 2 - minX * zoom);
    const y = Math.round(padTop + (ch - padTop - padBottom - bh * zoom) / 2 - minY * zoom);
    setViewport({ x, y, zoom: Number(zoom.toFixed(2)) });
  }, []);

  // Auto-fit initial preset once mounted
  useEffect(() => {
    const id = window.requestAnimationFrame(() => fitGraphToCanvas(CIRCUIT_PRESETS[0].graph));
    return () => window.cancelAnimationFrame(id);
  }, [fitGraphToCanvas]);

  const handleSelectPreset = useCallback(
    (presetId: string) => {
      const preset = CIRCUIT_PRESETS.find((p) => p.id === presetId);
      if (!preset) return;
      const cloned = cloneCircuitGraph(preset.graph);
      cloned.execution.mode = circuit.execution.mode;
      commitGraph(cloned);
      setActivePresetId(preset.id);
      setSelectedNodeId(cloned.nodes[0]?.id ?? null);
      setSelectedEdgeId(null);
      setRunResult(null);
      setNodeStatuses({});
      setNodeTelemetry({});
      setActiveEdgeIds(new Set());
      window.requestAnimationFrame(() => fitGraphToCanvas(cloned));
    },
    [circuit.execution.mode, commitGraph, fitGraphToCanvas],
  );

  const handleCheckLocalHealth = useCallback(async () => {
    setLocalHealth((prev) => ({ ...prev, checking: true }));
    const res = await checkLocalEngineHealth();
    setLocalHealth(res);
  }, []);

  const handleChangeEngineMode = useCallback(
    (mode: ExecutionEngineMode) => {
      setCircuit((prev) => ({ ...prev, execution: { ...prev.execution, mode } }));
      if (mode === 'local_python') {
        handleCheckLocalHealth();
      }
    },
    [handleCheckLocalHealth],
  );

  // Add block (click + or drag-drop)
  const handleAddBlock = useCallback(
    (blockType: string, customPos?: { x: number; y: number }) => {
      const existingIds = new Set(circuit.nodes.map((n) => n.id));
      let pos = customPos;
      if (!pos) {
        const el = canvasContainerRef.current;
        const cw = el?.clientWidth ?? 800;
        const ch = el?.clientHeight ?? 520;
        const offset = (circuit.nodes.length % 5) * 28;
        pos = {
          x: (cw / 2 - viewport.x) / viewport.zoom - NODE_WIDTH / 2 + offset,
          y: (ch / 2 - viewport.y) / viewport.zoom - 60 + offset,
        };
      }
      const created = createNodeFromBlock(blockType, pos, existingIds);
      if (!created) return;
      const next: CircuitGraphV1 = {
        ...circuit,
        nodes: [...circuit.nodes, created],
      };
      commitGraph(next);
      setActivePresetId(null);
      setSelectedNodeId(created.id);
      setSelectedEdgeId(null);
      setRightTab('inspector');
      setDrawerCollapsed(false);
    },
    [circuit, commitGraph, viewport],
  );

  const handleMoveNodePreview = useCallback((nodeId: string, pos: { x: number; y: number }) => {
    setCircuit((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => (n.id === nodeId ? { ...n, position: pos } : n)),
    }));
  }, []);

  const handleCommitNodeMove = useCallback(
    (nodeId: string, startPos: { x: number; y: number }, endPos: { x: number; y: number }) => {
      // Save pre-drag snapshot into Undo stack so dragging is a single clean undoable step
      setPast((p) => [
        ...p.slice(-(MAX_HISTORY - 1)),
        {
          ...circuit,
          nodes: circuit.nodes.map((n) => (n.id === nodeId ? { ...n, position: startPos } : n)),
        },
      ]);
      setFuture([]);
      setCircuit((prev) => ({
        ...prev,
        nodes: prev.nodes.map((n) => (n.id === nodeId ? { ...n, position: endPos } : n)),
      }));
    },
    [circuit],
  );

  const handleConnectPorts = useCallback(
    (sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string) => {
      const evalRes = evaluatePortConnection(
        circuit,
        sourceNodeId,
        sourcePortId,
        targetNodeId,
        targetPortId,
      );
      if (!evalRes.compatible) return;

      const sNode = circuit.nodes.find((n) => n.id === sourceNodeId);
      const sPort = sNode ? BLOCK_MAP[sNode.blockType]?.outputs.find((p) => p.id === sourcePortId) : undefined;
      if (!sPort) return;

      const filteredEdges = evalRes.replacesEdgeId
        ? circuit.edges.filter((e) => e.id !== evalRes.replacesEdgeId)
        : circuit.edges;

      const newEdge: CircuitEdge = {
        id: `e_${Date.now()}_${Math.floor(Math.random() * 999)}`,
        source: sourceNodeId,
        sourcePort: sourcePortId,
        target: targetNodeId,
        targetPort: targetPortId,
        signalType: sPort.signalType,
        kind: evalRes.kind,
        label: evalRes.kind === 'feedback' ? '↻ DMTA Round t+1 Update' : undefined,
      };

      const next: CircuitGraphV1 = {
        ...circuit,
        execution:
          evalRes.kind === 'feedback' && circuit.execution.dmtaRounds < 3
            ? { ...circuit.execution, dmtaRounds: 5 }
            : circuit.execution,
        edges: [...filteredEdges, newEdge],
      };
      commitGraph(next);
      setActivePresetId(null);
      setSelectedEdgeId(newEdge.id);
      setSelectedNodeId(null);
      if (evalRes.kind === 'feedback') {
        showToast('↻ Connected as DMTA Feedback Loop (5 rounds)');
      }
    },
    [circuit, commitGraph, showToast],
  );

  const handleUpdateNodeParam = useCallback(
    (nodeId: string, key: string, value: number | string | boolean) => {
      const next: CircuitGraphV1 = {
        ...circuit,
        nodes: circuit.nodes.map((n) =>
          n.id === nodeId ? { ...n, params: { ...n.params, [key]: value } } : n,
        ),
      };
      commitGraph(next);
    },
    [circuit, commitGraph],
  );

  const handleToggleNodeDisabled = useCallback(
    (nodeId: string) => {
      commitGraph({
        ...circuit,
        nodes: circuit.nodes.map((n) => (n.id === nodeId ? { ...n, disabled: !n.disabled } : n)),
      });
    },
    [circuit, commitGraph],
  );

  const handleDuplicateNode = useCallback(
    (nodeId: string) => {
      const orig = circuit.nodes.find((n) => n.id === nodeId);
      if (!orig) return;
      const existingIds = new Set(circuit.nodes.map((n) => n.id));
      const dup = createNodeFromBlock(
        orig.blockType,
        { x: orig.position.x + 36, y: orig.position.y + 36 },
        existingIds,
      );
      if (!dup) return;
      dup.params = { ...orig.params };
      commitGraph({ ...circuit, nodes: [...circuit.nodes, dup] });
      setSelectedNodeId(dup.id);
      setSelectedEdgeId(null);
    },
    [circuit, commitGraph],
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      commitGraph({
        ...circuit,
        nodes: circuit.nodes.filter((n) => n.id !== nodeId),
        edges: circuit.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      });
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
    },
    [circuit, commitGraph, selectedNodeId],
  );

  const handleDeleteEdge = useCallback(
    (edgeId: string) => {
      commitGraph({
        ...circuit,
        edges: circuit.edges.filter((e) => e.id !== edgeId),
      });
      if (selectedEdgeId === edgeId) setSelectedEdgeId(null);
    },
    [circuit, commitGraph, selectedEdgeId],
  );

  const handleAutoLayout = useCallback(() => {
    const laidOut = autoLayoutCircuit(circuit);
    commitGraph(laidOut);
    window.requestAnimationFrame(() => fitGraphToCanvas(laidOut));
    showToast('Auto-arranged circuit into clean topological layers');
  }, [circuit, commitGraph, fitGraphToCanvas, showToast]);

  const handleFocusNode = useCallback(
    (nodeId: string) => {
      const node = circuit.nodes.find((n) => n.id === nodeId);
      if (!node) return;
      setSelectedNodeId(nodeId);
      setSelectedEdgeId(null);
      const el = canvasContainerRef.current;
      if (el) {
        const cx = el.clientWidth / 2;
        const cy = el.clientHeight / 2;
        setViewport((vp) => ({
          ...vp,
          x: Math.round(cx - (node.position.x + NODE_WIDTH / 2) * vp.zoom),
          y: Math.round(cy - (node.position.y + 80) * vp.zoom),
        }));
      }
    },
    [circuit.nodes],
  );

  // Animated Wave Execution (Browser Sim OR Local Python Engine)
  const handleRunCircuit = useCallback(async () => {
    if (isRunning || circuit.nodes.length === 0) return;

    if (!validation.valid) {
      setRightTab('inspector');
      setDrawerCollapsed(false);
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      showToast('Resolve highlighted signal errors before running');
      return;
    }

    setIsRunning(true);
    setNodeStatuses({});
    setNodeTelemetry({});
    setActiveEdgeIds(new Set());

    try {
      let response: RunCircuitResponse;
      if (circuit.execution.mode === 'local_python') {
        try {
          response = await runCircuitOnLocalEngine(circuit);
        } catch {
          showToast('Local :8765 offline — falling back to Browser Simulation');
          response = await simulateCircuitInBrowser(circuit, molecules, v4Benchmarks);
        }
      } else {
        response = await simulateCircuitInBrowser(circuit, molecules, v4Benchmarks);
      }

      // Animate signal wave across topological stages!
      const latestRound = Math.max(1, response.roundsCompleted);
      const roundSteps = response.steps.filter((s) => s.round === latestRound);
      for (const step of roundSteps) {
        setNodeStatuses((prev) => ({ ...prev, [step.nodeId]: 'running' }));
        const outgoingIds = circuit.edges
          .filter((e) => e.source === step.nodeId)
          .map((e) => e.id);
        setActiveEdgeIds(new Set(outgoingIds));
        await new Promise((r) => window.setTimeout(r, 115));
        setNodeStatuses((prev) => ({
          ...prev,
          [step.nodeId]: step.status === 'error' ? 'error' : 'ok',
        }));
        setNodeTelemetry((prev) => ({ ...prev, [step.nodeId]: step }));
      }

      // Briefly illuminate DMTA feedback loop if present
      if (validation.hasFeedbackLoop) {
        setActiveEdgeIds(new Set(validation.feedbackEdges));
        await new Promise((r) => window.setTimeout(r, 420));
      }

      setActiveEdgeIds(new Set());
      setRunResult(response);
      setRightTab('telemetry');
      setDrawerCollapsed(false);
      showToast(
        `Completed ${response.roundsCompleted}r in ${response.totalDurationMs} ms · SHA ${response.provenanceSha256.slice(0, 8)}`,
      );
    } finally {
      setIsRunning(false);
    }
  }, [circuit, isRunning, molecules, showToast, v4Benchmarks, validation]);

  const handleExportJson = useCallback(() => {
    const slug = circuit.name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
    downloadTextFile(`${slug || 'geneuclidean_circuit'}.json`, exportCircuitJson(circuit), 'application/json');
    showToast('Exported CircuitGraphV1 JSON');
  }, [circuit, showToast]);

  const handleImportJson = useCallback(
    (rawText: string) => {
      const parsed = parseCircuitJson(rawText);
      if (parsed.error || !parsed.graph) {
        showToast(parsed.error ?? 'Invalid JSON');
        return;
      }
      commitGraph(parsed.graph);
      setActivePresetId(parsed.graph.metadata?.presetId ?? null);
      setSelectedNodeId(parsed.graph.nodes[0]?.id ?? null);
      setSelectedEdgeId(null);
      window.requestAnimationFrame(() => fitGraphToCanvas(parsed.graph!));
      showToast(`Imported "${parsed.graph.name}"`);
    },
    [commitGraph, fitGraphToCanvas, showToast],
  );

  // Global Keyboard Shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement;
      const mod = e.metaKey || e.ctrlKey;

      if (mod && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setDockCollapsed(false);
        window.requestAnimationFrame(() => searchInputRef.current?.focus());
        return;
      }
      if (mod && e.key === 'Enter') {
        e.preventDefault();
        handleRunCircuit();
        return;
      }
      if (inInput) return;

      if (mod && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      } else if ((mod && e.shiftKey && e.key.toLowerCase() === 'z') || (mod && e.key.toLowerCase() === 'y')) {
        e.preventDefault();
        handleRedo();
      } else if (mod && e.key.toLowerCase() === 'd' && selectedNodeId) {
        e.preventDefault();
        handleDuplicateNode(selectedNodeId);
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && (selectedNodeId || selectedEdgeId)) {
        e.preventDefault();
        if (selectedNodeId) handleDeleteNode(selectedNodeId);
        else if (selectedEdgeId) handleDeleteEdge(selectedEdgeId);
      } else if (e.key === '0') {
        e.preventDefault();
        fitGraphToCanvas(circuit);
      } else if (e.key.toLowerCase() === 'l') {
        e.preventDefault();
        handleAutoLayout();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    circuit,
    fitGraphToCanvas,
    handleAutoLayout,
    handleDeleteEdge,
    handleDeleteNode,
    handleDuplicateNode,
    handleRedo,
    handleRunCircuit,
    handleUndo,
    selectedEdgeId,
    selectedNodeId,
  ]);

  return (
    <div className="studio-circuit-workbench w-full flex flex-col overflow-hidden">
      {/* Top Contextual Preset & Execution Bar */}
      <PresetBar
        presets={CIRCUIT_PRESETS}
        activePresetId={activePresetId}
        onSelectPreset={handleSelectPreset}
        validation={validation}
        nodeCount={circuit.nodes.length}
        edgeCount={circuit.edges.length}
        dmtaRounds={circuit.execution.dmtaRounds}
        engineMode={circuit.execution.mode}
        onChangeEngineMode={handleChangeEngineMode}
        localHealth={localHealth}
        onRecheckLocal={handleCheckLocalHealth}
        onImportJson={handleImportJson}
        onExportJson={handleExportJson}
        onRunCircuit={handleRunCircuit}
        onFocusFirstIssue={() => {
          const firstIssue = validation.diagnostics.find((d) => d.nodeId);
          if (firstIssue?.nodeId) {
            handleFocusNode(firstIssue.nodeId);
          } else {
            setSelectedNodeId(null);
            setSelectedEdgeId(null);
          }
          setRightTab('inspector');
          setDrawerCollapsed(false);
        }}
        isRunning={isRunning}
      />

      {/* Main 3-Pane macOS Sequoia / Xcode Workspace */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Collapsible Block Library Dock */}
        <BlockPalette
          collapsed={dockCollapsed}
          onToggleCollapse={() => setDockCollapsed((c) => !c)}
          onAddBlock={(bt) => handleAddBlock(bt)}
          searchInputRef={searchInputRef}
          theme={theme}
        />

        {/* Center Infinite Pan/Zoom Spatial Breadboard Canvas */}
        <Canvas
          circuit={circuit}
          validation={validation}
          theme={theme}
          viewport={viewport}
          onChangeViewport={setViewport}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onSelectNode={setSelectedNodeId}
          onSelectEdge={setSelectedEdgeId}
          onMoveNodePreview={handleMoveNodePreview}
          onCommitNodeMove={handleCommitNodeMove}
          onConnectPorts={handleConnectPorts}
          onWiringRejected={(reason) => showToast(reason, 'error')}
          onDropNewBlock={(bt, pos) => handleAddBlock(bt, pos)}
          onFitView={() => fitGraphToCanvas(circuit)}
          onAutoLayout={handleAutoLayout}
          onUndo={handleUndo}
          onRedo={handleRedo}
          canUndo={past.length > 0}
          canRedo={future.length > 0}
          onDuplicateSelected={() => selectedNodeId && handleDuplicateNode(selectedNodeId)}
          onDeleteSelected={() => {
            if (selectedNodeId) handleDeleteNode(selectedNodeId);
            else if (selectedEdgeId) handleDeleteEdge(selectedEdgeId);
          }}
          onLoadDefaultPreset={() => handleSelectPreset(CIRCUIT_PRESETS[0].id)}
          nodeStatuses={nodeStatuses}
          nodeTelemetry={nodeTelemetry}
          activeEdgeIds={activeEdgeIds}
          isRunning={isRunning}
          toastMessage={toastMessage}
          canvasContainerRef={canvasContainerRef}
        />

        {/* Right Collapsible 3-Tab Inspector / Live Python SDK / Run Telemetry Drawer */}
        {drawerCollapsed ? (
          <aside className="w-11 shrink-0 h-full circuit-glass border-l border-[var(--circuit-node-border)] flex flex-col items-center py-2.5 gap-2 z-20">
            <button
              type="button"
              onClick={() => setDrawerCollapsed(false)}
              title="Expand Inspector Drawer"
              className="h-8 w-8 rounded-lg hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <PanelRightOpen className="h-4 w-4" />
            </button>
            <div className="w-6 h-[1px] bg-[var(--circuit-node-border)]" />
            {(
              [
                ['inspector', Sliders, 'Inspector'],
                ['python', Code2, 'Python SDK'],
                ['telemetry', Activity, 'Run Telemetry'],
              ] as const
            ).map(([id, Icon, label]) => (
              <button
                key={id}
                type="button"
                title={label}
                onClick={() => {
                  setRightTab(id);
                  setDrawerCollapsed(false);
                }}
                className={`h-8 w-8 rounded-lg flex items-center justify-center cursor-pointer ${
                  rightTab === id
                    ? 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300'
                    : 'text-[var(--text-muted)] hover:bg-slate-500/12'
                }`}
              >
                <Icon className="h-4 w-4" />
              </button>
            ))}
          </aside>
        ) : (
          <aside
            data-right-drawer="true"
            className="w-[320px] shrink-0 h-full circuit-glass border-l border-[var(--circuit-node-border)] flex flex-col z-20"
          >
            {/* Segmented Tab Switcher */}
            <div className="px-2 py-2 border-b border-[var(--circuit-node-border)] flex items-center justify-between gap-1 shrink-0">
              <div className="flex-1 grid grid-cols-[1fr_0.86fr_1.14fr] gap-0.5 p-0.5 rounded-lg bg-[var(--bg-elevated)]/85 border border-[var(--circuit-node-border)]">
                {(
                  [
                    ['inspector', 'Inspector', Sliders],
                    ['python', 'Python', Code2],
                    ['telemetry', 'Telemetry', Activity],
                  ] as const
                ).map(([id, label, Icon]) => {
                  const active = rightTab === id;
                  return (
                    <button
                      key={id}
                      type="button"
                      data-drawer-tab={id}
                      onClick={() => setRightTab(id)}
                      className={`py-1 px-1 rounded-md text-[10.5px] font-medium whitespace-nowrap flex items-center justify-center gap-1 transition-all cursor-pointer ${
                        active
                          ? 'bg-[var(--bg-surface)] text-[var(--text-primary)] shadow-xs font-semibold'
                          : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      <Icon className="h-3 w-3 shrink-0" />
                      <span>{label}</span>
                      {id === 'telemetry' && runResult && (
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>
              <button
                type="button"
                onClick={() => setDrawerCollapsed(true)}
                title="Collapse Inspector Drawer"
                className="h-7 w-7 rounded-md hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] shrink-0 cursor-pointer"
              >
                <PanelRightClose className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Drawer Body */}
            <div className="flex-1 min-h-0 overflow-y-auto">
              {rightTab === 'inspector' && (
                <InspectorPanel
                  circuit={circuit}
                  validation={validation}
                  selectedNode={selectedNode}
                  selectedEdge={selectedEdge}
                  onUpdateNodeParam={handleUpdateNodeParam}
                  onToggleNodeDisabled={handleToggleNodeDisabled}
                  onDuplicateNode={handleDuplicateNode}
                  onDeleteNode={handleDeleteNode}
                  onDeleteEdge={handleDeleteEdge}
                  onUpdateExecution={(patch) =>
                    commitGraph({
                      ...circuit,
                      execution: { ...circuit.execution, ...patch },
                    })
                  }
                  onFocusNode={handleFocusNode}
                  theme={theme}
                />
              )}

              {rightTab === 'python' && (
                <CodeExportDrawer
                  circuitName={circuit.name}
                  codegen={codegen}
                  selectedNodeId={selectedNodeId}
                  onExportJson={handleExportJson}
                />
              )}

              {rightTab === 'telemetry' && (
                <ExecutionTelemetry
                  runResult={runResult}
                  isRunning={isRunning}
                  moleculesIndex={molecules}
                  theme={theme}
                  onRunCircuit={handleRunCircuit}
                  onFocusNode={handleFocusNode}
                  onInspectIn3D={onInspectIn3D}
                />
              )}
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
