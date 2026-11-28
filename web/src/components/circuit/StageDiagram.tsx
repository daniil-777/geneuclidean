import { ArrowRight, ArrowUpRight, Box, Code2, Maximize, Minus, Plus, Workflow } from 'lucide-react';
import { useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { BLOCK_MAP, SIGNAL_META } from '../../lib/circuitCatalog';
import { getBlockImplementation, type CircuitStage, type ImplementationStep } from '../../lib/circuitStages';
import { layoutStageDiagram } from '../../lib/stageDiagram';
import type { CircuitEdge, CircuitNode } from '../../types/circuit';
import './stageDiagram.css';

interface StageDiagramProps {
  stage: CircuitStage;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
  nameOf?: (node: CircuitNode) => string;
  descriptionOf?: (node: CircuitNode) => string;
}

function blockName(node: CircuitNode) {
  const definition = Object.hasOwn(BLOCK_MAP, node.blockType) && BLOCK_MAP[node.blockType];
  return node.label || (definition && definition.shortName) || node.blockType;
}

function wireStyle(edge: CircuitEdge): CSSProperties {
  const signal = SIGNAL_META[edge.signalType];
  return { '--wire-light': signal?.lightHex ?? '#475569', '--wire-dark': signal?.darkHex ?? '#cbd5e1' } as CSSProperties;
}

function connectionLabel(edge: CircuitEdge, name: (id: string) => string) {
  return name(edge.source) + ' / ' + edge.sourcePort + ' → ' + name(edge.target) + ' / ' + edge.targetPort;
}

function Architecture({ stage, steps, onSelectNode, selectedNodeId, nameOf = blockName }: StageDiagramProps & {
  steps: readonly ImplementationStep[];
}) {
  const block = stage.nodes[0];
  return <section className="sd-frame" data-diagram-stage={stage.id} aria-label={stage.label + ' block diagram'}>
    <div className="sd-bar">
      <div className="sd-summary"><Code2 size={16} /><strong>Python architecture</strong></div>
      <button className="sd-architecture-block" type="button" data-block-id={block.id} data-node-id={block.id}
        data-diagram-node={block.id} aria-label={'Inspect ' + nameOf(block)} aria-pressed={block.id === selectedNodeId}
        onClick={() => onSelectNode(block.id)}>{nameOf(block)}<Box size={14} /></button>
    </div>
    <ol className="sd-steps sd-paper">{steps.map(step => <li key={step.source}>
      <a data-implementation-step={step.label}
        href={'https://github.com/daniil-777/geneuclidean/blob/main/' + step.source} target="_blank" rel="noreferrer">
        <strong>{step.label}<ArrowUpRight size={13} aria-hidden="true" /></strong>
        <span>{step.description}</span>
      </a>
    </li>)}</ol>
  </section>;
}

export function StageDiagram(props: StageDiagramProps) {
  const { stage } = props;
  const steps = stage.nodes.length === 1 && stage.internal.length === 0
    ? getBlockImplementation(stage.nodes[0].blockType) : [];
  if (steps.length) return <Architecture {...props} steps={steps} />;
  return <WiringDiagram key={stage.id} {...props} />;
}

function WiringDiagram({ stage, selectedNodeId, onSelectNode, nameOf = blockName, descriptionOf }: StageDiagramProps) {
  const viewport = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [zoomOverride, setZoom] = useState<number | null>(null);
  const [chosenId, chooseWire] = useState<string | null>(stage.internal[0]?.id ?? null);
  const [hoverId, hoverWire] = useState<string | null>(null);
  const markerId = useId();
  useLayoutEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const measure = () => setWidth(element.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const nodes = useMemo(() => new Map(stage.nodes.map(n => [n.id, n])), [stage.nodes]);
  const edges = useMemo(() => new Map(stage.internal.map(e => [e.id, e])), [stage.internal]);
  const wireChoices = useMemo(() => {
    const counts = new Map<string, number>();
    const candidateNames = stage.internal.map(edge => {
      const signal = SIGNAL_META[edge.signalType]?.label ?? edge.signalType;
      const family = edge.kind === 'feedback' ? 'Feedback · ' + signal : signal;
      const target = nodes.get(edge.target);
      const destination = family + ' → ' + (target ? nameOf(target) : edge.target);
      for (const key of [family, destination]) counts.set(key, (counts.get(key) ?? 0) + 1);
      return { edge, family, destination };
    });
    return candidateNames.map(({ edge, family, destination }) => {
      if (counts.get(family) === 1) return { edge, label: family };
      if (counts.get(destination) === 1) return { edge, label: destination };
      return { edge, label: destination + ' / ' + edge.targetPort + ' · ' + edge.id };
    });
  }, [stage.internal, nodes, nameOf]);
  const horizontal = useMemo(() => layoutStageDiagram(stage.nodes, stage.internal, 'horizontal'), [stage]);
  const vertical = useMemo(() => layoutStageDiagram(stage.nodes, stage.internal, 'vertical'), [stage]);
  const layout = horizontal.width + 16 <= width ? horizontal : vertical;
  const maxFit = layout === horizontal && width >= 820 ? 1.25 : 1;
  const fitZoom = width ? Math.max(.75, Math.min(maxFit, (width - 12) / layout.width)) : 1;
  const zoom = zoomOverride ?? fitZoom;
  const selectedWire = edges.get(hoverId ?? '') ?? edges.get(chosenId ?? '');
  const straightFlow = layout.edges.every(edge => edge.points.length === 2);
  const name = (id: string) => { const node = nodes.get(id); return node ? nameOf(node) : id; };
  function zoomBy(delta: number) {
    setZoom(Math.max(.75, Math.min(2, Math.round((zoom + delta) * 100) / 100)));
  }
  function fit() {
    setZoom(null);
    viewport.current?.scrollTo({ left: 0, top: 0 });
  }
  function choose(id: string) {
    chooseWire(id);
    hoverWire(null);
  }
  const count = (value: number, singular: string) => value + ' ' + singular + (value === 1 ? '' : 's');

  return <section className="sd-frame" data-diagram-stage={stage.id} aria-label={stage.label + ' block diagram'}>
    <div className="sd-bar">
      <div className="sd-summary"><Workflow size={16} /><span>{count(stage.nodes.length, 'block')}</span>
        <span className="sd-counter-dot">·</span><span>{count(stage.internal.length, 'wire')}</span></div>
      <div className="sd-zoom" role="group" aria-label="Diagram zoom">
        <button type="button" aria-label="Zoom out" title="Zoom out" onClick={() => zoomBy(-.25)} disabled={zoom <= .75}><Minus /></button>
        <output aria-label="Zoom level">{Math.round(zoom * 100)}%</output>
        <button type="button" aria-label="Zoom in" title="Zoom in" onClick={() => zoomBy(.25)} disabled={zoom >= 2}><Plus /></button>
        <button type="button" aria-label="Fit diagram" title="Fit diagram" onClick={fit}><Maximize /></button>
      </div>
    </div>
    <div ref={viewport} className="sd-viewport sd-paper" role="region" tabIndex={0} aria-label={stage.label + ' diagram workspace'}>
      <div className="sd-size" style={{ width: layout.width * zoom, height: layout.height * zoom }}>
        <div className="sd-board" style={{ width: layout.width, height: layout.height, transform: 'scale(' + zoom + ')' }}>
          <svg className="sd-layer sd-wires" width={layout.width} height={layout.height} aria-label="Diagram wires">
            <defs><marker id={markerId} viewBox="0 0 12 10" refX="13" refY="5" markerWidth="12" markerHeight="10"
              markerUnits="userSpaceOnUse" orient="auto"><path d="M 1 2 L 7 5 L 1 8" fill="none" stroke="context-stroke" strokeWidth="1.8" /></marker></defs>
            {layout.edges.map(edge => {
              const actual = edges.get(edge.id)!;
              const label = connectionLabel(actual, name);
              const isActive = selectedWire?.id === edge.id;
              return <g key={edge.id} style={wireStyle(actual)} className="sd-wire-group" data-active={isActive || undefined}
                data-feedback={edge.kind === 'feedback' || undefined}
                onMouseEnter={() => hoverWire(edge.id)} onMouseLeave={() => hoverWire(null)}>
                <path className="sd-wire-hit" d={edge.hitPath} aria-hidden="true" onClick={() => choose(edge.id)}><title>{label}</title></path>
                <path d={edge.path} className="sd-wire" data-wire-id={edge.id} data-source={edge.source} data-target={edge.target}
                  data-source-port={edge.sourcePort} data-target-port={edge.targetPort} markerEnd={'url(#' + markerId + ')'}
                  role="button" aria-label={'Inspect connection ' + label} aria-pressed={chosenId === edge.id} tabIndex={0}
                  onFocus={() => hoverWire(edge.id)} onBlur={() => hoverWire(null)} onClick={() => choose(edge.id)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); choose(edge.id); }
                  }}><title>{label}</title></path>
              </g>;
            })}
          </svg>
          {layout.nodes.map(position => {
            const node = nodes.get(position.id)!;
            const description = descriptionOf?.(node) || node.blockType;
            return <button key={node.id} type="button" className="sd-node" data-node-id={node.id} data-block-id={node.id}
              data-diagram-node={node.id} data-disabled={node.disabled || undefined} aria-pressed={selectedNodeId === node.id}
              aria-label={'Inspect ' + nameOf(node)} title={nameOf(node) + ' · ' + description}
              style={{ left: position.x, top: position.y, width: position.width, height: position.height }}
              onClick={() => onSelectNode(node.id)}>
              <strong><Box size={14} aria-hidden="true" /><span>{nameOf(node)}</span></strong>
              <small>{node.disabled && 'Disabled · '}{description}</small>
            </button>;
          })}
          <svg className="sd-layer sd-ports" width={layout.width} height={layout.height} aria-hidden="true">
            {layout.nodes.flatMap(node => node.ports.map(port => {
              const actual = edges.get(port.edgeId)!;
              return <circle key={node.id + '-' + port.edgeId + '-' + port.direction} cx={port.x} cy={port.y}
                r={selectedWire?.id === port.edgeId ? 4.7 : 3.8} className="sd-port" style={wireStyle(actual)}
                data-node-id={node.id} data-port-id={port.portId} data-port-direction={port.direction} data-edge-id={port.edgeId}
                data-side={port.side} onMouseEnter={() => hoverWire(port.edgeId)} onMouseLeave={() => hoverWire(null)}
                onClick={() => choose(port.edgeId)}>
                <title>{name(node.id) + ' / ' + port.portId}</title>
              </circle>;
            }))}
          </svg>
          {selectedWire && straightFlow && layout.nodes.flatMap(node => node.ports
            .filter(port => port.edgeId === selectedWire.id)
            .map(port => {
              const rightOfWire = port.x >= node.x + node.width / 2;
              const horizontalFlow = layout === horizontal;
              const left = horizontalFlow ? node.x + 8 : rightOfWire ? port.x + 9 : node.x;
              const top = horizontalFlow ? node.y + node.height + 1 : port.y + (port.direction === 'output' ? 3 : -16);
              const available = horizontalFlow ? node.width - 16 : rightOfWire ? node.x + node.width - left : port.x - left - 9;
              return <span key={node.id + port.direction} className="sd-port-caption" aria-hidden="true" style={{
                left, top, width: available, textAlign: !horizontalFlow && !rightOfWire ? 'right' : 'left',
              }}>{port.direction === 'output' ? 'out' : 'in'} · {port.portId}</span>;
            }))}
        </div>
      </div>
    </div>
    {stage.internal.length > 0 && <div className="sd-footer">
      <div className="sd-wire-picker" role="group" aria-label="Inspect a wire">
        {wireChoices.map(({ edge, label }) => <button type="button" key={edge.id} data-wire-choice={edge.id} style={wireStyle(edge)}
          aria-pressed={chosenId === edge.id} aria-label={'Trace ' + connectionLabel(edge, name)}
          onMouseEnter={() => hoverWire(edge.id)} onMouseLeave={() => hoverWire(null)} onFocus={() => hoverWire(edge.id)}
          onBlur={() => hoverWire(null)} onClick={() => choose(edge.id)}>
          <i aria-hidden="true" />{label}
        </button>)}
      </div>
      <div className="sd-wire-detail" data-wire-detail={selectedWire?.id ?? ''} aria-live="polite" aria-atomic="true">
        {selectedWire ? <>
          <span><strong>{name(selectedWire.source)}</strong><code>{selectedWire.sourcePort}</code></span>
          <ArrowRight size={15} aria-label="to" />
          <span><strong>{name(selectedWire.target)}</strong><code>{selectedWire.targetPort}</code></span>
        </> : <p>Select a wire to inspect its ports.</p>}
      </div>
    </div>}
  </section>;
}
