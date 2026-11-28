import {
  ArrowDown, ArrowLeft, ArrowRight, ArrowUpRight, Atom, Boxes, Braces,
  ChevronDown, FileOutput, GitBranch, Layers3, Repeat2,
  Scan, Search, SlidersHorizontal, Workflow, type LucideIcon,
} from 'lucide-react';
import { Fragment, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { BLOCK_MAP, SIGNAL_META } from '../../lib/circuitCatalog';
import { getBlockImplementation, projectCircuitStages, type CircuitStage, type StageId } from '../../lib/circuitStages';
import type { CircuitEdge, CircuitGraphV1, CircuitNode, ParamSpec, RunCircuitResponse } from '../../types/circuit';
import './stageEditor.css';

interface StageEditorProps {
  circuit: CircuitGraphV1;
  onUpdateNodeParam: (nodeId: string, key: string, value: string | number | boolean) => void;
  onOpenCanvas: (nodeId?: string) => void;
  result?: RunCircuitResponse | null;
}
type UpdateParam = StageEditorProps['onUpdateNodeParam'];

const icons: Record<StageId, LucideIcon> = {
  input: Scan, encoder: Layers3, decoder: GitBranch, smiles: Braces,
  screening: Search, learning: Repeat2, physics: Atom, optimization: SlidersHorizontal,
  output: FileOutput, other: Boxes,
};

const keyParams = [
  'target_id', 'on_target_id', 'n_conformers', 'pocket_cutoff', 'num_queries',
  'num_samples', 'decode_mode', 'temperature', 'num_layers', 'hidden_dim',
  'w_on', 'w_off', 'rl_objective', 'lora_rank', 'exhaustiveness',
];

const blockRoles = new Map<string, string>([
  ['protein_pocket', 'Loads the atoms and coordinates of a binding site.'],
  ['multiconformer_ensemble', 'Supplies several pocket shapes for the target and an optional comparison protein.'],
  ['painn_se3', 'Combines features from neighboring atoms to encode the pocket.'],
  ['cartformer_se3', 'Encodes the 3D arrangement of pocket atoms.'],
  ['nlp_perceiver_bridge', 'Compresses atom features into tokens for the decoder.'],
  ['nlp_rope_gqa_lm', 'Reads pocket tokens and proposes molecular strings.'],
  ['pocket_ensemble_cfg', 'Scores candidates across target and comparison pockets.'],
  ['nlp_lora_rl_aligner', 'Uses candidate rewards to guide alignment of the molecular model.'],
  ['safe_gpt', 'Samples a sequence of molecular fragments and decodes it to SMILES.'],
]);

function definitionFor(node: CircuitNode) {
  return Object.hasOwn(BLOCK_MAP, node.blockType) ? BLOCK_MAP[node.blockType] : undefined;
}

function blockName(node: CircuitNode) {
  return node.label || definitionFor(node)?.shortName || node.blockType;
}

function SignalArrow() {
  return <span className="stage-signal-arrow" aria-hidden="true"><ArrowRight /><ArrowDown /></span>;
}

function StageCard({ stage, onOpen }: { stage: CircuitStage; onOpen: () => void }) {
  const Icon = icons[stage.id];
  const count = stage.nodes.length;
  const disabled = stage.nodes.filter(node => node.disabled).length;
  const descriptionId = useId();
  return <button type="button" className="stage-card" data-stage-id={stage.id}
    aria-label={'Open ' + stage.label} aria-describedby={descriptionId} onClick={onOpen}>
    <span className="stage-card-top"><span className="stage-icon"><Icon /></span><ArrowUpRight className="stage-enter-icon" /></span>
    <strong>{stage.label}</strong><span className="stage-description" id={descriptionId}>{stage.description}</span>
    <span className="stage-card-count">{stage.id === 'smiles' ? 'Decoder output' : count + (count === 1 ? ' block' : ' blocks')}
      {disabled > 0 && ' · ' + disabled + ' disabled'}</span>
  </button>;
}

function KeyParameter({ spec, node, onUpdate }: { spec: ParamSpec; node: CircuitNode; onUpdate: UpdateParam }) {
  const id = useId();
  const value = node.params[spec.key] ?? spec.default;
  return <div className="stage-param">
    <label htmlFor={id}>{spec.label}</label>
    {spec.type === 'number' && spec.min !== undefined && spec.max !== undefined ?
      <div className="stage-range"><input id={id} data-param-key={spec.key} type="range" value={Number(value)}
        min={spec.min} max={spec.max} step={spec.step ?? 1} aria-valuetext={value + (spec.unit ? ' ' + spec.unit : '')}
        onChange={e => onUpdate(node.id, spec.key, Number(e.currentTarget.value))} />
        <output htmlFor={id}>{value}{spec.unit && ' ' + spec.unit}</output></div>
      : spec.type === 'boolean' ?
        <input id={id} data-param-key={spec.key} type="checkbox" checked={value === true} onChange={e => onUpdate(node.id, spec.key, e.currentTarget.checked)} />
      : spec.type === 'select' ?
        <select id={id} data-param-key={spec.key} value={String(value)} onChange={e => {
          const option = spec.options?.find(o => String(o) === e.currentTarget.value);
          if (option !== undefined) onUpdate(node.id, spec.key, option);
        }}>{spec.options?.map(v => <option value={String(v)} key={v}>{v}</option>)}</select>
      : <input id={id} data-param-key={spec.key} type={spec.type === 'number' ? 'number' : 'text'} value={String(value)} step={spec.step}
          min={spec.min} max={spec.max} onChange={e => {
            if (spec.type !== 'number') onUpdate(node.id, spec.key, e.currentTarget.value);
            else if (e.currentTarget.value !== '' && e.currentTarget.validity.valid)
              onUpdate(node.id, spec.key, e.currentTarget.valueAsNumber);
          }} />}
  </div>;
}

function BlockSettings({ node, onUpdate, onOpenCanvas }: {
  node: CircuitNode; onUpdate: UpdateParam; onOpenCanvas: StageEditorProps['onOpenCanvas'];
}) {
  const def = definitionFor(node);
  const ordered = [...new Set([...keyParams, ...(def?.params ?? []).map(p => p.key)])];
  const specs = ordered.map(k => def?.params.find(s => s.key === k)).filter((s): s is ParamSpec => !!s).slice(0, 2);
  const steps = getBlockImplementation(node.blockType);
  return <div className="stage-inspect" aria-label={'Inspecting ' + blockName(node)}>
    {steps.length > 0 && <section className="stage-architecture">
      <h3>Python architecture</h3>
      <ol>{steps.map(step => <li key={step.label}><a href={'https://github.com/daniil-777/geneuclidean/blob/main/' + step.source}
        target="_blank" rel="noreferrer">{step.label}<ArrowUpRight size={13} /></a><p>{step.description}</p></li>)}</ol>
    </section>}
    <section className="stage-settings">
      <div className="stage-settings-head"><div><h3>Key parameters</h3><p>{blockName(node)}</p></div>
        <button type="button" onClick={() => onOpenCanvas(node.id)}>Edit in full canvas <ArrowUpRight size={15} /></button></div>
      {specs.length ? <div className="stage-param-grid">{specs.map(spec =>
        <KeyParameter key={spec.key} spec={spec} node={node} onUpdate={onUpdate} />)}</div> :
        <p>This block has no catalog parameters.</p>}
    </section>
  </div>;
}

function ConnectionRows({ edges, nodes, nameOf, onInspect }: {
  edges: CircuitEdge[]; nodes: Map<string, CircuitNode>; nameOf: (node: CircuitNode) => string; onInspect: (node: CircuitNode) => void;
}) {
  function endpoint(id: string, port: string, direction: 'inputs' | 'outputs') {
    const node = nodes.get(id);
    const label = node && definitionFor(node)?.[direction].find(p => p.id === port)?.label;
    const name = node ? nameOf(node) : id;
    const content = <><strong>{name}</strong><code title={label ?? port}>{port}</code></>;
    return node ? <button type="button" className="stage-endpoint" onClick={() => onInspect(node)}
      aria-label={'Inspect ' + name + ', ' + (direction === 'outputs' ? 'output' : 'input') + ' port ' + port}>{content}</button> :
      <span className="stage-endpoint">{content}</span>;
  }
  return <ul className="stage-connections">{edges.map(e =>
    <li key={e.id} data-connection-id={e.id}>
      {endpoint(e.source, e.sourcePort, 'outputs')}
      <span className="stage-edge-kind">{e.kind === 'feedback' ? <><Repeat2 size={14} /> Feedback</> :
        <><ArrowRight size={15} /><span>{SIGNAL_META[e.signalType]?.shortLabel ?? e.signalType}</span></>}</span>
      {endpoint(e.target, e.targetPort, 'inputs')}
    </li>)}</ul>;
}

export function StageEditor({ circuit, onUpdateNodeParam, onOpenCanvas, result }: StageEditorProps) {
  const model = useMemo(() => projectCircuitStages(circuit), [circuit]);
  const nodes = useMemo(() => new Map(circuit.nodes.map(n => [n.id, n])), [circuit.nodes]);
  const nameCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const node of circuit.nodes) {
      const name = blockName(node);
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    return counts;
  }, [circuit.nodes]);
  function nameOf(node: CircuitNode) {
    const name = blockName(node);
    return (nameCounts.get(name) ?? 0) > 1 ? name + ' · ' + node.id : name;
  }
  const [stageId, setStageId] = useState<StageId | null>(null);
  const [nodeId, setNodeId] = useState<string | null>(null);
  const [extraOpen, setExtraOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const title = useRef<HTMLHeadingElement>(null);
  const lastStage = useRef<StageId | null>(null);
  const overviewOrigin = useRef<StageId | null>(null);
  const stage = model.stages.find(s => s.id === stageId);
  const selected = stage?.nodes.find(n => n.id === nodeId);
  const adjacentForward = useMemo(() => {
    const linked = new Set<string>();
    if (!stage) return linked;
    const previous = new Map(stage.nodes.slice(1).map((node, i) => [node.id, stage.nodes[i].id]));
    for (const edge of stage.internal) {
      if (edge.kind === 'forward' && previous.get(edge.target) === edge.source) linked.add(edge.target);
    }
    return linked;
  }, [stage]);
  const open = (item: CircuitStage, id?: string) => {
    if (!stage) {
      overviewOrigin.current = item.id;
      if (model.additional.includes(item)) setExtraOpen(true);
    }
    setStageId(item.id);
    setNodeId(id ?? (item.nodes.length === 1 ? item.nodes[0].id : null));
  };
  function inspect(node: CircuitNode) {
    const group = model.stages.find(s => s.id === model.stageByNode.get(node.id));
    if (group) open(group, node.id);
  }
  useLayoutEffect(() => {
    if (stage && lastStage.current !== stage.id) title.current?.focus();
    if (!stage && lastStage.current && overviewOrigin.current) {
      root.current?.querySelector<HTMLButtonElement>('[data-stage-id="' + overviewOrigin.current + '"]')?.focus();
    }
    lastStage.current = stage?.id ?? null;
  }, [stage?.id]);

  const unresolved = model.unresolved.length > 0 && <details className="stage-unresolved">
    <summary>Unresolved connections <small>{model.unresolved.length}</small><ChevronDown size={16} /></summary>
    <p>A block at one or both ends of these wires is missing.</p>
    <ConnectionRows edges={model.unresolved} nodes={nodes} nameOf={nameOf} onInspect={inspect} />
  </details>;

  if (!model.stages.length) return <div className="stage-editor stage-empty">
    <Workflow size={35} /><h2>This circuit is empty</h2><p>Add a block to start your signal path.</p>
    <button type="button" onClick={() => onOpenCanvas()}>Open full canvas <ArrowUpRight size={16} /></button>
    {unresolved}
  </div>;

  const stageLinks = (from: StageId, to: StageId) =>
    model.links.some(link => link.source === from && link.target === to && link.kind !== 'feedback');
  const StageIcon = stage && icons[stage.id];
  const count = (n: number) => n + (n === 1 ? ' connection' : ' connections');
  return <div className="stage-editor" ref={root}>
    {!stage ? <>
      <section className="stage-path" aria-labelledby="stage-path-title">
        <div className="stage-section-heading"><h2 id="stage-path-title">Signal path</h2>
          <p>Choose a stage to see its blocks and connections.</p></div>
        <div className={'stage-flow' + (model.primary.length > 4 ? ' stage-many' : '')}>
          {model.primary.map((s, i) => <Fragment key={s.id}>
            {i > 0 && (stageLinks(model.primary[i - 1].id, s.id) ? <SignalArrow /> : <span className="stage-gap" />)}
            <StageCard stage={s} onOpen={() => open(s)} />
          </Fragment>)}
        </div>
      </section>
      {model.additional.length > 0 && <details className="stage-additional" open={extraOpen}
        onToggle={e => setExtraOpen(e.currentTarget.open)}>
        <summary><GitBranch size={18} /><span>Supporting stages</span><small>{model.additional.length}</small><ChevronDown size={18}/></summary>
        <div>{model.additional.map(s => <StageCard key={s.id} stage={s} onOpen={() => open(s)} />)}</div>
      </details>}
      {unresolved}
    </> : <>
      <button type="button" className="stage-back" onClick={() => {setStageId(null); setNodeId(null);}}><ArrowLeft size={16}/> Back to overview</button>
      <section className="stage-detail" data-open-stage={stage.id} key={stage.id}>
        <div className="stage-detail-heading"><span className="stage-icon">{StageIcon && <StageIcon/>}</span>
          <div><h2 ref={title} tabIndex={-1}>{stage.label}</h2><p>{stage.description}</p></div></div>
        {stage.id === 'smiles' ?
          <div className="stage-products">
            <h3>Decoder output ports</h3>
            <div>{stage.products.map(product => {const producer = nodes.get(product.nodeId); return producer &&
              <button type="button" key={product.nodeId + '.' + product.portId} onClick={() => inspect(producer)}>
                <span><strong>{nameOf(producer)}</strong><code>{product.portId}</code></span><ArrowUpRight size={16}/></button>;})}</div>
            <h3>{(result?.engineMode ?? circuit.execution.mode) === 'browser_sim' ? 'Browser preview · example molecules' : 'Circuit run output'}</h3>
            {result?.engineMode === 'browser_sim' && <p>The browser preview uses examples from the studio’s reference collection.</p>}
            {result?.engineMode === 'local_python' && result.steps.some(s => s.status === 'simulated_fallback') &&
              <p>This run used simulated steps. The list shows circuit-level results.</p>}
            {result?.molecules.length ? <ul>{result.molecules.map((m,i) => <li key={m.id + '-' + i}><strong>{m.name || m.id}</strong><code>{m.smiles}</code></li>)}</ul> :
              <p>{result ? 'This run returned no molecules.' : circuit.execution.mode === 'browser_sim'
                ? 'Run preview to inspect reference SMILES.' : 'Run the circuit to inspect its molecule strings.'}</p>}
          </div> : <>
            <h3>Blocks</h3>
            <div className={'stage-blocks' + (stage.nodes.length > 4 ? ' stage-many' : '')}>
              {stage.nodes.map((node, i) => <Fragment key={node.id}>
                {i > 0 && (adjacentForward.has(node.id) ? <SignalArrow/> : <span className="stage-gap"/>)}
                <button type="button" className="stage-block" title={node.blockType} aria-label={'Inspect ' + nameOf(node)}
                  aria-pressed={selected?.id === node.id} data-block-id={node.id} onClick={() => setNodeId(node.id)}>
                  <strong>{nameOf(node)}</strong>
                  <small>{node.disabled ? 'Disabled · ' : ''}{blockRoles.get(node.blockType) ?? definitionFor(node)?.summary ?? node.blockType}</small>
                </button>
              </Fragment>)}
            </div>
            {selected && <BlockSettings node={selected} onUpdate={onUpdateNodeParam} onOpenCanvas={onOpenCanvas}/>}
            <section className="stage-wires" aria-label="Connections"><h3>Connections</h3>
              {stage.internal.length > 0 && <ConnectionRows edges={stage.internal} nodes={nodes} nameOf={nameOf} onInspect={inspect}/>}
              {stage.incoming.length + stage.outgoing.length > 0 && <details open={stage.internal.length === 0 || undefined}>
                <summary><span>Boundary connections</span><small>{count(stage.incoming.length + stage.outgoing.length)}</small><ChevronDown size={16}/></summary>
                <ConnectionRows edges={[...stage.incoming, ...stage.outgoing]} nodes={nodes} nameOf={nameOf} onInspect={inspect}/>
              </details>}
              {!stage.internal.length && !stage.incoming.length && !stage.outgoing.length && <p>There are no wires to these blocks.</p>}
            </section>
          </>}
      </section>
    </>}
  </div>;
}
