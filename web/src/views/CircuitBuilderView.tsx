import {
  Activity,
  Code2,
  PanelRightClose,
  PanelRightOpen,
  Sliders,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { BlockPalette } from '../components/circuit/BlockPalette';
import { CircuitCanvas } from '../components/circuit/CircuitCanvas';
import { CodeExportDrawer } from '../components/circuit/CodeExportDrawer';
import { ExecutionTelemetry } from '../components/circuit/ExecutionTelemetry';
import { InspectorPanel } from '../components/circuit/InspectorPanel';
import { PresetBar } from '../components/circuit/PresetBar';
import {
  BLOCK_CATALOG_MAP,
  CIRCUIT_PRESETS,
  cloneCircuitGraph,
  createDefaultParams,
} from '../lib/circuitCatalog';
import {
  DEFAULT_LOCAL_STUDIO_URL,
  autoLayoutCircuit,
  canConnectPorts,
  checkLocalEngineHealth,
  exportCircuitJson,
  generatePythonSdk,
  parseCircuitJson,
  runCircuitOnLocalEngine,
  simulateCircuitInBrowser,
  validateCircuit,
} from '../lib/circuitEngine';
import { downloadTextFile } from '../lib/data';
import type {
  CircuitEdge,
  CircuitGraphV1,
  CircuitNode,
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

function nextUniqueId(prefix: string, existingIds: Set<string>): string {
  let idx = 1;
  while (existingIds.has(`${prefix}_${idx}`)) {
    idx += 1;
  }
  return `${prefix}_${idx}`;
}

export function CircuitBuilderView({
  molecules,
  v4Benchmarks,
  theme,
  onInspectIn3D,
}: CircuitBuilderViewProps) {
  const [circuit, setCircuit] = useState<CircuitGraphV1>(() =>
    cloneCircuitGraph(CIRCUIT_PRESETS[0].graph),
  );
  const [activePresetId, setActivePresetId] = useState<string>(CIRCUIT_PRESETS[0].id);

  // 50-step Undo / Redo stacks
  const [undoStack, setUndoStack] = useState<CircuitGraphV1[]>([]);
  const [redoStack, setRedoStack] = useState<CircuitGraphV1[]>([]);

  // Selection & Workspace Panes
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [rightTab, setRightTab] = useState<RightDrawerTab>('inspector');

  // Canvas Viewport & Imperative Signals
  const [viewport, setViewport] = useState({ x: 32, y: 40, zoom: 0.78 });
  const [fitViewSignal, setFitViewSignal] = useState(1);
  const [focusNodeSignal, setFocusNodeSignal] = useState<{ nodeId: string; seq: number } | null>(
    null,
  );
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  // Execution & Live Animation State
  const [isRunning, setIsRunning] = useState(false);
  const [runResult, setRunResult] = useState<RunCircuitResponse | null>(null);
  const [nodeStatusMap, setNodeStatusMap] = useState<Record<string, NodeRuntimeStatus>>({});
  const [nodeTelemetryMap, setNodeTelemetryMap] = useState<
    Record<string, ExecutionStepTelemetry>
  >({});
  const [activeEdgeIds, setActiveEdgeIds] = useState<Set<string>>(new Set());
  const runGenerationRef = useRef(0);

  // Local Python Engine Health
  const [localHealth, setLocalHealth] = useState<LocalEngineHealth>({
    reachable: false,
    checking: false,
  });

  // Ephemeral Toast
  const [toast, setToast] = useState<{ text: string; tone: 'info' | 'feedback' | 'error' } | null>(
    null,
  );
  const toastTimerRef = useRef<number | null>(null);

  const showToast = useCallback((text: string, tone: 'info' | 'feedback' | 'error' = 'info') => {
    setToast({ text, tone });
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => setToast(null), 3200);
  }, []);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
    };
  }, []);

  // Derived Validation & Live Python SDK Codegen
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

  // Undo-aware mutation helper
  const commitCircuit = useCallback((updater: (prev: CircuitGraphV1) => CircuitGraphV1) => {
    setCircuit((prev) => {
      const next = updater(prev);
      if (next === prev) return prev;
      setUndoStack((stack) => [...stack.slice(-(MAX_HISTORY - 1)), prev]);
      setRedoStack([]);
      return next;
    });
  }, []);

  const handleUndo = useCallback(() => {
    setUndoStack((stack) => {
      if (stack.length === 0) return stack;
      const previous = stack[stack.length - 1];
      setCircuit((curr) => {
        setRedoStack((r) => [...r.slice(-(MAX_HISTORY - 1)), curr]);
        return previous;
      });
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      return stack.slice(0, -1);
    });
  }, []);

  const handleRedo = useCallback(() => {
    setRedoStack((stack) => {
      if (stack.length === 0) return stack;
      const next = stack[stack.length - 1];
      setCircuit((curr) => {
        setUndoStack((u) => [...u.slice(-(MAX_HISTORY - 1)), curr]);
        return next;
      });
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      return stack.slice(0, -1);
    });
  }, []);

  // Probe Local Python Engine when user toggles to Local :8765
  const probeLocalEngine = useCallback(async () => {
    setLocalHealth((prev) => ({ ...prev, checking: true }));
    const health = await checkLocalEngineHealth(DEFAULT_LOCAL_STUDIO_URL);
    setLocalHealth(health);
    if (!health.reachable) {
      showToast('Local engine offline — run `geneuclidean studio --port 8765` (Browser Sim fallback active)', 'info');
    } else {
      showToast(`Connected to Local GenEuclidean Python v${health.version ?? '5.1'}`, 'info');
    }
  }, [showToast]);

  const handleChangeEngineMode = useCallback(
    (mode: ExecutionEngineMode) => {
      setCircuit((prev) => ({
        ...prev,
        execution: { ...prev.execution, mode },
      }));
      if (mode === 'local_python') {
        void probeLocalEngine();
      }
    },
    [probeLocalEngine],
  );

  // Preset Switching
  const handleSelectPreset = useCallback((presetId: string) => {
    const preset = CIRCUIT_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    runGenerationRef.current += 1;
    setIsRunning(false);
    setActivePresetId(presetId);
    setCircuit((prev) => {
      setUndoStack((u) => [...u.slice(-(MAX_HISTORY - 1)), prev]);
      setRedoStack([]);
      const cloned = cloneCircuitGraph(preset.graph);
      cloned.execution.mode = prev.execution.mode;
      return cloned;
    });
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setRunResult(null);
    setNodeStatusMap({});
    setNodeTelemetryMap({});
    setActiveEdgeIds(new Set());
    setFitViewSignal((s) => s + 1);
  }, []);

  // Block Insertion (click + or drop onto breadboard)
  const handleAddBlockAt = useCallback(
    (blockType: string, worldPos?: { x: number; y: number }) => {
      if (!BLOCK_CATALOG_MAP[blockType]) return;
      let createdId = '';
      commitCircuit((prev) => {
        const existingIds = new Set(prev.nodes.map((n) => n.id));
        const shortSlug = blockType.split('_')[0];
        createdId = nextUniqueId(`n_${shortSlug}`, existingIds);

        let pos = worldPos;
        if (!pos) {
          if (prev.nodes.length === 0) {
            pos = { x: 180, y: 150 };
          } else {
            const maxX = Math.max(...prev.nodes.map((n) => n.position.x));
            const avgY =
              prev.nodes.reduce((acc, n) => acc + n.position.y, 0) / prev.nodes.length;
            pos = {
              x: Math.round(maxX + 272),
              y: Math.round(avgY + ((prev.nodes.length % 3) - 1) * 44),
            };
          }
        }

        const newNode: CircuitNode = {
          id: createdId,
          blockType,
          position: pos,
          params: createDefaultParams(blockType),
        };
        return { ...prev, nodes: [...prev.nodes, newNode] };
      });
      if (createdId) {
        setSelectedNodeId(createdId);
        setSelectedEdgeId(null);
        setRightTab('inspector');
        setRightCollapsed(false);
      }
    },
    [commitCircuit],
  );

  // Live 60 FPS Node Dragging (no Undo push until pointerup)
  const handleMoveNodeLive = useCallback((nodeId: string, position: { x: number; y: number }) => {
    setCircuit((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => (n.id === nodeId ? { ...n, position } : n)),
    }));
  }, []);

  const handleCommitNodeMove = useCallback(
    (nodeId: string, startPos: { x: number; y: number }, endPos: { x: number; y: number }) => {
      setCircuit((curr) => {
        const prevSnapshot: CircuitGraphV1 = {
          ...curr,
          nodes: curr.nodes.map((n) => (n.id === nodeId ? { ...n, position: startPos } : n)),
        };
        setUndoStack((u) => [...u.slice(-(MAX_HISTORY - 1)), prevSnapshot]);
        setRedoStack([]);
        return {
          ...curr,
          nodes: curr.nodes.map((n) => (n.id === nodeId ? { ...n, position: endPos } : n)),
        };
      });
    },
    [],
  );

  // Interactive Port-to-Port Wiring with Smart Auto-Feedback Promotion
  const handleConnectEdge = useCallback(
    (sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string) => {
      const check = canConnectPorts(circuit, sourceNodeId, sourcePortId, targetNodeId, targetPortId);
      if (!check.compatible) {
        if (check.remediation) showToast(check.remediation, 'error');
        return;
      }

      const srcNode = circuit.nodes.find((n) => n.id === sourceNodeId);
      const srcPort = srcNode
        ? BLOCK_CATALOG_MAP[srcNode.blockType]?.outputs.find((p) => p.id === sourcePortId)
        : null;
      const dstNode = circuit.nodes.find((n) => n.id === targetNodeId);
      const dstPort = dstNode
        ? BLOCK_CATALOG_MAP[dstNode.blockType]?.inputs.find((p) => p.id === targetPortId)
        : null;
      if (!srcPort || !dstPort) return;

      commitCircuit((prev) => {
        // Smoothly replace single-input edge if multi !== true
        const keptEdges =
          dstPort.multi === true
            ? prev.edges
            : prev.edges.filter(
                (e) => !(e.target === targetNodeId && e.targetPort === targetPortId),
              );
        const existingEdgeIds = new Set(keptEdges.map((e) => e.id));
        const newEdge: CircuitEdge = {
          id: nextUniqueId(check.wouldBeFeedback ? 'e_fb' : 'e', existingEdgeIds),
          source: sourceNodeId,
          sourcePort: sourcePortId,
          target: targetNodeId,
          targetPort: targetPortId,
          signalType: srcPort.signalType,
          kind: check.wouldBeFeedback ? 'feedback' : 'forward',
          label: check.wouldBeFeedback ? '↻ DMTA Round t+1 Update' : undefined,
        };
        return {
          ...prev,
          edges: [...keptEdges, newEdge],
          execution:
            check.wouldBeFeedback && prev.execution.dmtaRounds < 3
              ? { ...prev.execution, dmtaRounds: 5 }
              : prev.execution,
        };
      });

      if (check.wouldBeFeedback) {
        showToast('↻ Connected as DMTA Feedback Loop (5 rounds enabled)', 'feedback');
      }
    },
    [circuit, commitCircuit, showToast],
  );

  // Node & Edge Mutations
  const handleUpdateNodeParam = useCallback(
    (nodeId: string, key: string, value: number | string | boolean) => {
      commitCircuit((prev) => ({
        ...prev,
        nodes: prev.nodes.map((n) =>
          n.id === nodeId ? { ...n, params: { ...n.params, [key]: value } } : n,
        ),
        execution:
          key === 'target_id' || key === 'on_target_id'
            ? { ...prev.execution, targetContext: String(value) }
            : prev.execution,
      }));
    },
    [commitCircuit],
  );

  const handleToggleNodeDisabled = useCallback(
    (nodeId: string) => {
      commitCircuit((prev) => ({
        ...prev,
        nodes: prev.nodes.map((n) => (n.id === nodeId ? { ...n, disabled: !n.disabled } : n)),
      }));
    },
    [commitCircuit],
  );

  const handleDuplicateNode = useCallback(
    (nodeId: string) => {
      let duplicatedId = '';
      commitCircuit((prev) => {
        const orig = prev.nodes.find((n) => n.id === nodeId);
        if (!orig) return prev;
        const existingIds = new Set(prev.nodes.map((n) => n.id));
        duplicatedId = nextUniqueId(`${orig.id}_copy`, existingIds);
        const copy: CircuitNode = {
          ...orig,
          id: duplicatedId,
          position: { x: orig.position.x + 36, y: orig.position.y + 44 },
          params: { ...orig.params },
        };
        return { ...prev, nodes: [...prev.nodes, copy] };
      });
      if (duplicatedId) {
        setSelectedNodeId(duplicatedId);
        setSelectedEdgeId(null);
      }
    },
    [commitCircuit],
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      commitCircuit((prev) => ({
        ...prev,
        nodes: prev.nodes.filter((n) => n.id !== nodeId),
        edges: prev.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
      }));
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
    },
    [commitCircuit, selectedNodeId],
  );

  const handleDeleteEdge = useCallback(
    (edgeId: string) => {
      commitCircuit((prev) => ({
        ...prev,
        edges: prev.edges.filter((e) => e.id !== edgeId),
      }));
      if (selectedEdgeId === edgeId) setSelectedEdgeId(null);
    },
    [commitCircuit, selectedEdgeId],
  );

  const handleToggleEdgeFeedback = useCallback(
    (edgeId: string) => {
      commitCircuit((prev) => ({
        ...prev,
        edges: prev.edges.map((e) =>
          e.id === edgeId
            ? {
                ...e,
                kind: e.kind === 'feedback' ? 'forward' : 'feedback',
                label: e.kind === 'feedback' ? undefined : '↻ DMTA Round t+1 Update',
              }
            : e,
        ),
      }));
    },
    [commitCircuit],
  );

  const handleDeleteSelected = useCallback(() => {
    if (selectedNodeId) handleDeleteNode(selectedNodeId);
    else if (selectedEdgeId) handleDeleteEdge(selectedEdgeId);
  }, [selectedNodeId, selectedEdgeId, handleDeleteNode, handleDeleteEdge]);

  const handleAutoLayout = useCallback(() => {
    commitCircuit((prev) => autoLayoutCircuit(prev));
    setFitViewSignal((s) => s + 1);
    showToast('Applied clean topological left-to-right layout', 'info');
  }, [commitCircuit, showToast]);

  const handleFocusNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    setSelectedEdgeId(null);
    setFocusNodeSignal((prev) => ({ nodeId, seq: (prev?.seq ?? 0) + 1 }));
  }, []);

  const handleFocusFirstIssue = useCallback(() => {
    const first =
      validation.diagnostics.find((d) => d.severity === 'error') ??
      validation.diagnostics.find((d) => d.severity === 'warning');
    setRightTab('inspector');
    setRightCollapsed(false);
    if (first?.nodeId) {
      handleFocusNode(first.nodeId);
    } else {
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
    }
  }, [validation.diagnostics, handleFocusNode]);

  // JSON Export / Import
  const handleExportJson = useCallback(() => {
    const slug =
      circuit.name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_|_$/g, '') || 'circuit';
    downloadTextFile(`geneuclidean_${slug}.json`, exportCircuitJson(circuit), 'application/json');
    showToast(`Exported geneuclidean_${slug}.json`, 'info');
  }, [circuit, showToast]);

  const handleImportJson = useCallback(
    (text: string) => {
      const parsed = parseCircuitJson(text);
      if (parsed.error || !parsed.graph) {
        showToast(parsed.error ?? 'Invalid circuit JSON file.', 'error');
        return;
      }
      const imported = parsed.graph;
      commitCircuit(() => imported);
      setActivePresetId(imported.metadata?.presetId ?? '');
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setRunResult(null);
      setNodeStatusMap({});
      setNodeTelemetryMap({});
      setFitViewSignal((s) => s + 1);
      showToast(`Imported "${imported.name}" (${imported.nodes.length} blocks)`, 'info');
    },
    [commitCircuit, showToast],
  );

  // Execute Circuit (Dual-Mode: Browser Sim or Local Python :8765) with Wave Animation
  const handleRunCircuit = useCallback(async () => {
    if (isRunning || circuit.nodes.length === 0) return;

    if (!validation.valid) {
      handleFocusFirstIssue();
      const firstErr = validation.diagnostics.find((d) => d.severity === 'error');
      showToast(firstErr?.message ?? 'Fix circuit validation issues before running', 'error');
      return;
    }

    const genId = ++runGenerationRef.current;
    setIsRunning(true);
    setRightTab('telemetry');
    setRightCollapsed(false);
    setNodeStatusMap({});
    setNodeTelemetryMap({});
    setActiveEdgeIds(new Set());

    // Fetch / simulate execution result first so we have exact real telemetry to animate
    let response: RunCircuitResponse;
    if (circuit.execution.mode === 'local_python') {
      try {
        response = await runCircuitOnLocalEngine(circuit, DEFAULT_LOCAL_STUDIO_URL);
        setLocalHealth((prev) => ({ ...prev, reachable: true, checking: false }));
      } catch {
        setLocalHealth((prev) => ({
          ...prev,
          reachable: false,
          checking: false,
          error: 'Local server unreachable at 127.0.0.1:8765',
        }));
        showToast('Local :8765 unreachable — ran deterministic Browser Simulation', 'info');
        response = await simulateCircuitInBrowser(circuit, molecules, v4Benchmarks);
      }
    } else {
      response = await simulateCircuitInBrowser(circuit, molecules, v4Benchmarks);
    }

    if (runGenerationRef.current !== genId) return;

    // Animate luminous signal wave through topological order (+ feedback loop pulse if DMTA!)
    const topo = validation.topologicalOrder;
    const finalRoundSteps = response.steps.filter((s) => s.round === response.roundsCompleted);
    const stepByNode = new Map(finalRoundSteps.map((s) => [s.nodeId, s]));
    const feedbackEdgeIds = circuit.edges.filter((e) => e.kind === 'feedback').map((e) => e.id);

    for (let i = 0; i < topo.length; i++) {
      if (runGenerationRef.current !== genId) return;
      const nodeId = topo[i];
      const incomingIds = circuit.edges
        .filter((e) => e.target === nodeId && e.kind === 'forward')
        .map((e) => e.id);
      const outgoingIds = circuit.edges
        .filter((e) => e.source === nodeId && e.kind === 'forward')
        .map((e) => e.id);

      setNodeStatusMap((prev) => ({ ...prev, [nodeId]: 'running' }));
      setActiveEdgeIds(new Set([...incomingIds, ...outgoingIds]));

      await new Promise((r) => window.setTimeout(r, 95));
      if (runGenerationRef.current !== genId) return;

      const tele = stepByNode.get(nodeId);
      setNodeStatusMap((prev) => ({
        ...prev,
        [nodeId]: tele?.status === 'error' ? 'error' : 'ok',
      }));
      if (tele) {
        setNodeTelemetryMap((prev) => ({ ...prev, [nodeId]: tele }));
      }
    }

    // Pulse DMTA Feedback Loop arcs briefly at the end of the wave
    if (feedbackEdgeIds.length > 0) {
      setActiveEdgeIds(new Set(feedbackEdgeIds));
      await new Promise((r) => window.setTimeout(r, 260));
      if (runGenerationRef.current !== genId) return;
    }

    setActiveEdgeIds(new Set());
    setRunResult(response);
    setIsRunning(false);
  }, [
    isRunning,
    circuit,
    validation,
    molecules,
    v4Benchmarks,
    handleFocusFirstIssue,
    showToast,
  ]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const inInput =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT');
      const meta = e.metaKey || e.ctrlKey;

      if (meta && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setLeftCollapsed(false);
        requestAnimationFrame(() => searchInputRef.current?.focus());
        return;
      }

      if (meta && e.key === 'Enter') {
        e.preventDefault();
        void handleRunCircuit();
        return;
      }

      if (meta && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }

      if (meta && e.key.toLowerCase() === 'd' && selectedNodeId) {
        e.preventDefault();
        handleDuplicateNode(selectedNodeId);
        return;
      }

      if (inInput) return;

      if (e.key === 'Backspace' || e.key === 'Delete') {
        if (selectedNodeId || selectedEdgeId) {
          e.preventDefault();
          handleDeleteSelected();
        }
      } else if (e.key === '0') {
        e.preventDefault();
        setFitViewSignal((s) => s + 1);
      } else if (e.key.toLowerCase() === 'l') {
        e.preventDefault();
        handleAutoLayout();
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    handleRunCircuit,
    handleUndo,
    handleRedo,
    handleDuplicateNode,
    handleDeleteSelected,
    handleAutoLayout,
    selectedNodeId,
    selectedEdgeId,
  ]);

  return (
    <div className="h-[calc(100vh-56px)] w-full flex flex-col overflow-hidden bg-[var(--circuit-canvas-bg)]">
      {/* Top Contextual Apple Preset & Execution Bar */}
      <PresetBar
        presets={CIRCUIT_PRESETS}
        activePresetId={activePresetId}
        onSelectPreset={handleSelectPreset}
        nodeCount={circuit.nodes.length}
        edgeCount={circuit.edges.length}
        dmtaRounds={circuit.execution.dmtaRounds}
        validation={validation}
        onFocusFirstIssue={handleFocusFirstIssue}
        engineMode={circuit.execution.mode}
        onChangeEngineMode={handleChangeEngineMode}
        localHealth={localHealth}
        onImportJson={handleImportJson}
        onExportJson={handleExportJson}
        onRunCircuit={() => void handleRunCircuit()}
        isRunning={isRunning}
      />

      {/* 3-Pane Full-Height Studio Workspace */}
      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* Left Pane: Collapsible 36-Block Library Dock */}
        <BlockPalette
          theme={theme}
          collapsed={leftCollapsed}
          onToggleCollapse={() => setLeftCollapsed((c) => !c)}
          onAddBlock={(blockType) => handleAddBlockAt(blockType)}
          searchInputRef={searchInputRef}
        />

        {/* Center Pane: Infinite Pan/Zoom Breadboard Canvas */}
        <CircuitCanvas
          nodes={circuit.nodes}
          edges={circuit.edges}
          theme={theme}
          viewport={viewport}
          onViewportChange={setViewport}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onSelectNode={(id) => {
            setSelectedNodeId(id);
            if (id) setRightTab('inspector');
          }}
          onSelectEdge={(id) => {
            setSelectedEdgeId(id);
            if (id) setRightTab('inspector');
          }}
          onMoveNodeLive={handleMoveNodeLive}
          onCommitNodeMove={handleCommitNodeMove}
          onConnectEdge={handleConnectEdge}
          onDropNewBlock={(blockType, worldPos) => handleAddBlockAt(blockType, worldPos)}
          validation={validation}
          nodeStatusMap={nodeStatusMap}
          nodeTelemetryMap={nodeTelemetryMap}
          activeEdgeIds={activeEdgeIds}
          isRunning={isRunning}
          canUndo={undoStack.length > 0}
          canRedo={redoStack.length > 0}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onAutoLayout={handleAutoLayout}
          onDuplicateSelected={() => selectedNodeId && handleDuplicateNode(selectedNodeId)}
          onDeleteSelected={handleDeleteSelected}
          onQuickAddBlock={(blockType) => handleAddBlockAt(blockType)}
          onQuickLoadPreset={handleSelectPreset}
          fitViewSignal={fitViewSignal}
          focusNodeSignal={focusNodeSignal}
          toastMessage={toast}
          onDismissToast={() => setToast(null)}
        />

        {/* Right Pane: Tabbed Inspector / Live Python SDK / Run Telemetry Drawer */}
        {rightCollapsed ? (
          <aside className="w-11 shrink-0 circuit-glass border-l border-[var(--border-subtle)] flex flex-col items-center py-2.5 gap-2 z-10">
            <button
              type="button"
              onClick={() => setRightCollapsed(false)}
              title="Expand Inspector Drawer"
              aria-label="Expand Inspector Drawer"
              className="h-8 w-8 rounded-lg border border-[var(--border-subtle)] bg-[var(--circuit-node-bg)] hover:border-[var(--border-strong)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <PanelRightOpen className="h-4 w-4" />
            </button>
            <div className="w-5 h-px bg-[var(--border-subtle)] my-0.5" />
            {(
              [
                ['inspector', Sliders, 'Inspector'],
                ['python', Code2, 'Python SDK'],
                ['telemetry', Activity, 'Run Telemetry'],
              ] as const
            ).map(([id, Icon, lbl]) => (
              <button
                key={id}
                type="button"
                onClick={() => {
                  setRightTab(id);
                  setRightCollapsed(false);
                }}
                title={lbl}
                className={`h-8 w-8 rounded-lg flex items-center justify-center transition-colors cursor-pointer ${
                  rightTab === id
                    ? 'bg-cyan-500/15 text-cyan-600 dark:text-cyan-300'
                    : 'text-[var(--text-muted)] hover:bg-slate-500/10 hover:text-[var(--text-primary)]'
                }`}
              >
                <Icon className="h-4 w-4" />
              </button>
            ))}
          </aside>
        ) : (
          <aside className="w-[348px] shrink-0 circuit-glass border-l border-[var(--border-subtle)] flex flex-col h-full overflow-hidden z-10">
            {/* Segmented Apple 3-Tab Switcher */}
            <div className="px-3 py-2 border-b border-[var(--border-subtle)] flex items-center justify-between gap-2 shrink-0">
              <div className="flex-1 grid grid-cols-3 p-0.5 rounded-xl bg-slate-500/10 dark:bg-white/[0.05] border border-[var(--border-subtle)]">
                {(
                  [
                    ['inspector', 'Inspector', Sliders],
                    ['python', 'Python SDK', Code2],
                    ['telemetry', 'Telemetry', Activity],
                  ] as const
                ).map(([id, label, Icon]) => {
                  const active = rightTab === id;
                  return (
                    <button
                      key={id}
                      type="button"
                      onClick={() => setRightTab(id)}
                      className={`py-1 px-1.5 rounded-[9px] text-[11px] font-medium flex items-center justify-center gap-1 transition-all cursor-pointer ${
                        active
                          ? 'bg-[var(--circuit-node-bg)] text-[var(--text-primary)] shadow-sm border border-[var(--circuit-node-border)] font-semibold'
                          : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent'
                      }`}
                    >
                      <Icon className="h-3 w-3 shrink-0" />
                      <span className="truncate">{label}</span>
                      {id === 'telemetry' && runResult && (
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>

              <button
                type="button"
                onClick={() => setRightCollapsed(true)}
                title="Collapse Inspector Drawer"
                aria-label="Collapse Inspector Drawer"
                className="h-7 w-7 rounded-lg hover:bg-slate-500/12 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] shrink-0 cursor-pointer"
              >
                <PanelRightClose className="h-3.5 w-3.5" />
              </button>
            </div>

            {/* Active Right Tab Body */}
            <div className="flex-1 overflow-y-auto min-h-0">
              {rightTab === 'inspector' && (
                <InspectorPanel
                  circuit={circuit}
                  theme={theme}
                  selectedNode={selectedNode}
                  selectedEdge={selectedEdge}
                  validation={validation}
                  onUpdateNodeParam={handleUpdateNodeParam}
                  onToggleNodeDisabled={handleToggleNodeDisabled}
                  onDuplicateNode={handleDuplicateNode}
                  onDeleteNode={handleDeleteNode}
                  onDeleteEdge={handleDeleteEdge}
                  onToggleEdgeFeedback={handleToggleEdgeFeedback}
                  onUpdateExecution={(patch) =>
                    commitCircuit((prev) => ({
                      ...prev,
                      execution: { ...prev.execution, ...patch },
                    }))
                  }
                  onFocusNode={handleFocusNode}
                />
              )}

              {rightTab === 'python' && (
                <CodeExportDrawer
                  codegen={codegen}
                  circuitName={circuit.name}
                  onExportJson={handleExportJson}
                />
              )}

              {rightTab === 'telemetry' && (
                <ExecutionTelemetry
                  runResult={runResult}
                  isRunning={isRunning}
                  theme={theme}
                  moleculesIndex={molecules}
                  onRunCircuit={() => void handleRunCircuit()}
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
