import { ArrowDownToLine, Check, ChevronDown, ExternalLink, Link2, RotateCcw } from 'lucide-react';
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  defaultTwinInputs,
  leadProperties,
  simulateHumanTwin,
  twinRanges,
  type FormulationId,
  type LeadId,
  type TwinInputs,
} from '../lib/humanTwin';
import './humanTwin.css';

const formulations: { id: FormulationId; label: string }[] = [
  { id: 'rapid_equilibrium', label: 'Rapid equilibrium' },
  { id: 'faster_solid_release', label: 'Fast release' },
  { id: 'slower_solid_release', label: 'Slow release' },
  { id: 'gastric_to_intestine', label: 'Gastric transfer' },
];
const labels = Object.fromEntries(formulations.map(({ id, label }) => [id, label]));
const controls = {
  dissolutionPerH: { label: 'Intestinal dissolution', unit: '/h', ...twinRanges.dissolutionPerH, step: 0.05, decimals: 2 },
  transitPerH: { label: 'Intestinal transit', unit: '/h', ...twinRanges.transitPerH, step: 0.05, decimals: 2 },
  clearanceLPerH: { label: 'Liver intrinsic clearance', unit: 'L/h', ...twinRanges.clearanceLPerH, step: 1, decimals: 0 },
} as const;
type ControlKey = keyof typeof controls;
type Result = ReturnType<typeof simulateHumanTwin>;
type Compartment = 'brain' | 'plasma';

function readScenario() {
  const params = new URLSearchParams(window.location.hash.split('?')[1]);
  const lead: LeadId = params.get('lead') === 'DRD1_1' ? 'DRD1_1' : 'DRD1_v53_L1';
  const requested = formulations.find(({ id }) => id === params.get('form'));
  const inputs = defaultTwinInputs(lead, requested?.id ?? 'gastric_to_intestine');
  for (const [key, param] of [
    ['transitPerH', 'transit'], ['clearanceLPerH', 'cl'], ['dissolutionPerH', 'diss'],
  ] as const) {
    const raw = params.get(param);
    const value = Number(raw);
    const spec = controls[key];
    if (raw?.trim() && Number.isFinite(value) && value >= spec.min && value <= spec.max) {
      inputs[key] = Number((spec.min + Math.round((value - spec.min) / spec.step) * spec.step).toFixed(spec.decimals));
    }
  }
  return inputs;
}

function linkFor(inputs: TwinInputs) {
  const params = new URLSearchParams({
    lead: inputs.lead, form: inputs.formulation, transit: String(inputs.transitPerH),
    cl: String(inputs.clearanceLPerH), diss: String(inputs.dissolutionPerH),
  });
  const link = new URL(window.location.href);
  link.hash = 'human-twin?' + params.toString();
  return link.toString();
}

function maxTick(value: number) {
  const rough = Math.max(value, 0.001) / 4;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].find((n) => n >= rough / magnitude) ?? 10;
  return step * magnitude * 4;
}

function ExposureChart({ run, reference, formulation, curveLabel }: {
  run: Result; reference: Result; formulation: FormulationId; curveLabel: string;
}) {
  const [compartment, setCompartment] = useState<Compartment>('brain');
  const [observedAt, setObservedAt] = useState<number | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(680);
  const titleId = useId();
  const axisId = useId();
  const gradId = useId();
  useLayoutEffect(() => {
    const node = container.current;
    if (!node) return;
    setWidth(Math.max(180, node.getBoundingClientRect().width));
    const resize = new ResizeObserver(([entry]) => setWidth(Math.max(180, entry.contentRect.width)));
    resize.observe(node);
    return () => resize.disconnect();
  }, []);

  const compared = formulation !== 'rapid_equilibrium';
  const key = compartment === 'brain' ? 'brainNm' : 'plasmaNm';
  const { samples } = run;
  const maxValue = Math.max(...samples.map((s) => s[key]), ...(compared ? reference.samples.map((s) => s[key]) : [0]));
  const ceil = maxTick(maxValue);
  const height = width < 480 ? 190 : 218;
  const x0 = 39, x1 = width - 12, y0 = 12, y1 = height - 28;
  const x = (hour: number) => x0 + (x1 - x0) * hour / 48;
  const y = (nm: number) => y1 - (y1 - y0) * nm / ceil;
  const pathFor = (result: Result) => result.samples
    .map((s, i) => (i ? 'L' : 'M') + x(s.timeH).toFixed(2) + ',' + y(s[key]).toFixed(2)).join(' ');
  const path = pathFor(run);
  const sampleIndex = observedAt ?? samples.reduce((best, s, i) => s[key] > samples[best][key] ? i : best, 0);
  const sample = samples[sampleIndex];
  const peakName = compartment === 'brain' ? 'brain' : 'plasma';
  const chartTitle = 'Unbound ' + peakName + ' concentration';
  return <section className="ht-chart" aria-labelledby={titleId}>
    <div className="ht-chart-heading">
      <h2 id={titleId}>Exposure profile</h2>
      <div role="group" aria-label="Concentration compartment" className="ht-chart-toggle">
        {(['brain', 'plasma'] as const).map((part) => <button type="button" key={part}
          aria-pressed={compartment === part} onClick={() => { setCompartment(part); setObservedAt(null); }}>
          {part === 'brain' ? 'Brain' : 'Plasma'}
        </button>)}
      </div>
    </div>
    <div className="ht-plot" ref={container}>
      <div className="ht-axis-caption">{chartTitle} (nM)</div>
      <svg viewBox={'0 0 ' + width + ' ' + height} role="img" aria-labelledby={axisId}
        onPointerMove={(event) => {
          if (event.pointerType !== 'mouse') return;
          const pos = (event.clientX - event.currentTarget.getBoundingClientRect().left - x0) / (x1 - x0);
          setObservedAt(Math.max(0, Math.min(samples.length - 1, Math.round(pos * (samples.length - 1)))));
        }}>
        <title id={axisId}>{chartTitle} over 48 hours. Use the time control below for individual values.</title>
        <defs><linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--ht-trace)" stopOpacity=".19" />
          <stop offset="100%" stopColor="var(--ht-trace)" stopOpacity=".015" />
        </linearGradient></defs>
        {[0, 1, 2, 3, 4].map((n) => <g key={n}>
          <line x1={x0} x2={x1} y1={y(ceil * n / 4)} y2={y(ceil * n / 4)} className="ht-gridline" />
          <text x={x0 - 9} y={y(ceil * n / 4) + 4} textAnchor="end">{Number((ceil * n / 4).toPrecision(3))}</text>
        </g>)}
        {[0, 12, 24, 36, 48].map((hour) =>
          <text key={hour} x={x(hour)} y={height - 7} textAnchor={hour === 48 ? 'end' : 'middle'}>{hour} h</text>)}
        <path d={path + ' L' + x1 + ',' + y1 + ' L' + x0 + ',' + y1 + ' Z'} fill={'url(#' + gradId + ')'} />
        {compared && <path d={pathFor(reference)} className="ht-reference-line" />}
        <path d={path} className="ht-active-line" />
        <line x1={x(sample.timeH)} x2={x(sample.timeH)} y1={y0} y2={y1} className="ht-cursor" />
        <circle cx={x(sample.timeH)} cy={y(sample[key])} r="4.8" className="ht-cursor-point" />
      </svg>
    </div>
    <div className="ht-legend">
      <span><i />{curveLabel}</span>
      {compared && <span><i className="ht-dashes" />Rapid equilibrium</span>}
    </div>
    <div className="ht-inspect">
      <label htmlFor={axisId + '-time'}>Inspect time <strong>{sample.timeH.toFixed(1)} h</strong></label>
      <input id={axisId + '-time'} aria-label="Inspect time" type="range" min={0} max={48} step={0.5}
        value={sample.timeH} style={{ '--ht-filled': (sample.timeH / 48 * 100) + '%' } as CSSProperties} aria-valuetext={sample.timeH + ' hours, ' + sample[key].toFixed(2) + ' nanomolar'}
        onChange={(e) => setObservedAt(Math.round(Number(e.currentTarget.value) * 2))} />
      <output htmlFor={axisId + '-time'}>{sample[key].toFixed(2)} nM</output>
    </div>
  </section>;
}

function ParameterControl({ name, inputs, onChange }: {
  name: ControlKey; inputs: TwinInputs; onChange: (key: ControlKey, value: number) => void;
}) {
  const spec = controls[name];
  const id = useId();
  return <div className="ht-range">
    <div><label htmlFor={id}>{spec.label}</label>
      <output htmlFor={id}>{inputs[name].toFixed(spec.decimals)} <span>{spec.unit}</span></output></div>
    <input id={id} type="range" min={spec.min} max={spec.max} step={spec.step} value={inputs[name]}
      style={{ '--ht-filled': ((inputs[name] - spec.min) / (spec.max - spec.min) * 100) + '%' } as CSSProperties}
      aria-valuetext={inputs[name] + ' ' + spec.unit} onChange={(e) => onChange(name, Number(e.currentTarget.value))} />
  </div>;
}

export function HumanTwinView() {
  const [inputs, setInputs] = useState(readScenario);
  const [copied, setCopied] = useState(false);
  const [fallbackLink, setFallbackLink] = useState('');
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const fallbackInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const onHash = () => {
      if (window.location.hash.split('?')[0] === '#human-twin') setInputs(readScenario());
    };
    window.addEventListener('hashchange', onHash);
    return () => { window.removeEventListener('hashchange', onHash); clearTimeout(timer.current); };
  }, []);
  useEffect(() => { setCopied(false); setFallbackLink(''); }, [inputs]);
  useEffect(() => { fallbackInput.current?.select(); }, [fallbackLink]);

  const run = useMemo(() => simulateHumanTwin(inputs), [inputs]);
  const reference = useMemo(() => simulateHumanTwin({
    ...defaultTwinInputs(inputs.lead, 'rapid_equilibrium'), transitPerH: inputs.transitPerH,
    clearanceLPerH: inputs.clearanceLPerH,
  }), [inputs.lead, inputs.transitPerH, inputs.clearanceLPerH]);
  const candidate = leadProperties[inputs.lead];
  const customRelease = inputs.formulation !== 'rapid_equilibrium'
    && inputs.dissolutionPerH !== defaultTwinInputs(inputs.lead, inputs.formulation).dissolutionPerH;
  const curveLabel = customRelease
    ? (inputs.formulation === 'gastric_to_intestine' ? 'Gastric transfer' : 'Intestinal release')
      + ' · ' + inputs.dissolutionPerH.toFixed(2) + '/h'
    : labels[inputs.formulation];
  function applyEdit(next: TwinInputs) {
    // Only an explicit edit changes history. Mounts, hash navigation and Back
    // read the existing URL, so incoming scenario links stay authoritative.
    const link = linkFor(next);
    if (window.location.href !== link) {
      window.history.replaceState(window.history.state, '', link);
    }
    setInputs(next);
  }
  const change = (name: ControlKey, value: number) => applyEdit({ ...inputs, [name]: value });
  function pickFormulation(id: FormulationId) {
    if (id === inputs.formulation && !customRelease) return;
    applyEdit({ ...inputs, formulation: id, dissolutionPerH: defaultTwinInputs(inputs.lead, id).dissolutionPerH });
  }
  async function copyLink() {
    const link = linkFor(inputs);
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 2400);
    } catch { setFallbackLink(link); }
  }
  function download() {
    const report = {
      schemaVersion: 1, application: 'GenEuclidean Web Studio', scenarioUrl: linkFor(inputs), inputs,
      normalizedOralInputUmol: 1, durationH: 48, candidate, exposure: run,
      rapidEquilibriumComparison: reference,
      interpretation: 'Research scenario. Individual clinical safety, D1 engagement and tic response are undetermined.',
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = inputs.lead + '-' + inputs.formulation + '.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="ht-page">
    <header className="ht-header">
      <div><h1>Tourette simulator</h1>
        <p>Compare how an oral formulation changes a research compound’s exposure in the brain and plasma.</p></div>
      <div className="ht-actions">
        <button type="button" onClick={copyLink}>{copied ? <Check /> : <Link2 />}<span aria-live="polite">{copied ? 'Copied' : 'Copy link'}</span></button>
        <button type="button" onClick={download}><ArrowDownToLine /> Download JSON</button>
      </div>
    </header>
    {fallbackLink && <label className="ht-copy-fallback">Copy this scenario link
      <input ref={fallbackInput} readOnly value={fallbackLink} onFocus={(e) => e.currentTarget.select()} /></label>}
    <div className="ht-presets">
      <div className="ht-lead">
        <label htmlFor="ht-compound">Research compound</label>
        <div><select id="ht-compound" value={inputs.lead} onChange={(e) => applyEdit({
          ...inputs, lead: e.currentTarget.value as LeadId,
        })}><option value="DRD1_v53_L1">DRD1_v53_L1</option><option value="DRD1_1">DRD1_1</option></select><ChevronDown /></div>
      </div>
      <fieldset className="ht-formulations"><legend>Oral formulation</legend>
        <div>{formulations.map(({ id, label }) => <button key={id} type="button" aria-pressed={inputs.formulation === id && !customRelease}
          onClick={() => pickFormulation(id)}>{label}</button>)}</div>
      </fieldset>
    </div>
    <div className="ht-workspace">
      <div className="ht-run">
        <ExposureChart run={run} reference={reference} formulation={inputs.formulation} curveLabel={curveLabel} />
        <section className="ht-metrics" aria-label="Modeled unbound brain exposure">
          <div><span>Peak brain concentration</span><strong>{run.peakBrainNm.toFixed(2)} <small>nM</small></strong></div>
          <div><span>Time to brain peak</span><strong>{run.timeToPeakH.toFixed(2)} <small>h</small></strong></div>
          <div><span>Brain exposure over 48 h</span><strong>{run.brainAucNmH.toFixed(2)} <small>nM·h</small></strong></div>
        </section>
      </div>
      <section className="ht-params">
        <div className="ht-params-heading"><h2>Model parameters</h2>
          <button type="button" aria-label="Reset parameters" title="Reset parameters" onClick={() => applyEdit(defaultTwinInputs(inputs.lead, inputs.formulation))}><RotateCcw /></button></div>
        <p>Adjust a rate to recalculate exposure.</p>
        {inputs.formulation !== 'rapid_equilibrium' && <ParameterControl name="dissolutionPerH" inputs={inputs} onChange={change} />}
        <ParameterControl name="transitPerH" inputs={inputs} onChange={change} />
        <ParameterControl name="clearanceLPerH" inputs={inputs} onChange={change} />
        <div className="ht-constants">
          <div><span>Normalized oral input</span><strong>1 µmol</strong></div>
          <div><span>Simulation window</span><strong>48 h</strong></div>
        </div>
      </section>
    </div>
    <details className="ht-evidence">
      <summary>Assumptions and evidence <ChevronDown /></summary>
      <div>
        <p>Each run follows unchanged parent compound for 48 hours. Liver intrinsic clearance acts on unbound drug. Simulated peak and total exposure use the integrated trajectory. The chart shows half-hour samples.</p>
        <p>The candidate’s estimated intestinal solubility is {candidate.solubilityUm.toFixed(1)} µM and its estimated unbound fraction in plasma is {(candidate.fractionUnboundPlasma * 100).toFixed(1)}%. The comparison uses the same compound, transit and clearance with immediate equilibration in the intestine.</p>
        {inputs.formulation === 'gastric_to_intestine' && <p>The gastric case uses separate solid and liquid emptying rates derived from a mixed-meal study. Applying them to this compound is an assumption. The dissolution control changes release in the intestine.</p>}
        <div className="ht-links"><a href={'https://pubchem.ncbi.nlm.nih.gov/compound/' + candidate.pubchemCid} target="_blank" rel="noreferrer">Compound record <ExternalLink /></a>
          <a href="https://github.com/daniil-777/geneuclidean/blob/main/docs/tourette_human_twin.md" target="_blank" rel="noreferrer">Model and validation <ExternalLink /></a></div>
      </div>
    </details>
    <p className="ht-scope">Research simulation. This compound’s effect on tics and its safety in people remain unknown.</p>
  </div>;
}
