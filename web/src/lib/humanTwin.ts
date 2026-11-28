// Browser counterpart of human_twin.exposure.simulate_oral for the studio's
// buffered, partner-free research scenarios. All states except AUC are µmol.
import model from './humanTwinParameters.json' with { type: 'json' };

export type LeadId = keyof typeof model.leads;
export type FormulationId = keyof typeof model.formulations;

export interface TwinInputs {
  lead: LeadId;
  formulation: FormulationId;
  transitPerH: number;
  clearanceLPerH: number;
  dissolutionPerH: number;
}

export const leadProperties = model.leads;
export const twinRanges = {
  transitPerH: { min: 0.05, max: 1 },
  clearanceLPerH: { min: 1, max: 80 },
  dissolutionPerH: { min: 0.1, max: 4 },
} as const;

export function defaultTwinInputs(lead: LeadId, formulation: FormulationId): TwinInputs {
  if (!Object.hasOwn(leadProperties, lead) || !Object.hasOwn(model.formulations, formulation)) {
    throw new RangeError('Unknown candidate or formulation');
  }
  return {
    lead, formulation,
    transitPerH: model.disposition.gut_transit_per_h,
    clearanceLPerH: model.disposition.liver_unbound_clint_l_h,
    dissolutionPerH: model.formulations[formulation]?.dissolution_per_h
      ?? model.formulations.faster_solid_release.dissolution_per_h,
  };
}

export interface TwinSample {
  timeH: number;
  brainNm: number;
  plasmaNm: number;
  solidUmol: number;
  dissolvedUmol: number;
}

const gutLiquid = 0;
const gutSolid = 1;
const stomachLiquid = 2;
const stomachSolid = 3;
const central = 4;
const peripheral = 5;
const brain = 6;
const transitLoss = 7;
const firstPassLoss = 8;
const systemicLoss = 9;
const brainAuc = 10;
const states = 11;

// A 90-second RK4 step resolves the fastest rate in the bounded sliders;
// samples for display are every half-hour. Parity checks use the Python
// Radau solver independently, including at jointly perturbed rates.
const stepH = 1 / 40;
const reportEvery = 20;

function transfer(liquid: number, solid: number, capacity: number, release: number, precipitate: number) {
  const gap = capacity - Math.max(liquid, 0);
  return gap < 0 ? precipitate * gap : release * Math.min(Math.max(solid, 0), gap);
}

// Locate the interior maximum in a step with a positive-to-negative brain flux.
// Hermite interpolation uses concentrations and fluxes at both step boundaries.
function stepPeak(left: number, right: number, leftSlope: number, rightSlope: number) {
  const delta = right - left;
  const c = stepH * leftSlope;
  const b = 3 * delta - stepH * (2 * leftSlope + rightSlope);
  const a = stepH * (leftSlope + rightSlope) - 2 * delta;
  let lo = 0, hi = 1;
  for (let i = 0; i < 32; i++) {
    const mid = (lo + hi) / 2;
    if ((3 * a * mid + 2 * b) * mid + c > 0) lo = mid;
    else hi = mid;
  }
  const fraction = (lo + hi) / 2;
  return { offset: fraction * stepH, amount: ((a * fraction + b) * fraction + c) * fraction + left };
}

export function simulateHumanTwin(inputs: TwinInputs) {
  const { lead, formulation, transitPerH, clearanceLPerH, dissolutionPerH } = inputs;
  if (!Object.hasOwn(leadProperties, lead) || !Object.hasOwn(model.formulations, formulation)) {
    throw new RangeError('Unknown candidate or formulation');
  }
  for (const key of Object.keys(twinRanges) as (keyof typeof twinRanges)[]) {
    const value = inputs[key];
    const { min, max } = twinRanges[key];
    if (!Number.isFinite(value) || value < min || value > max) {
      throw new RangeError(`${key} must be between ${min} and ${max}`);
    }
  }
  const compound = leadProperties[lead];
  const p = model.disposition;
  const stage = model.gastric;
  const form = model.formulations[formulation];
  const hasStomach = formulation === 'gastric_to_intestine';
  const fu = compound.fractionUnboundPlasma;
  const unboundBloodCl = fu / p.blood_to_plasma_ratio * clearanceLPerH;
  const hepaticCl = p.liver_blood_flow_l_h * unboundBloodCl / (p.liver_blood_flow_l_h + unboundBloodCl);
  const systemicCl = hepaticCl * p.blood_to_plasma_ratio + p.other_plasma_clearance_l_h;
  const available = p.gut_escape_fraction * (1 - hepaticCl / p.liver_blood_flow_l_h);
  const gutCapacity = model.intestinalMedium.volume_l * compound.solubilityUm;
  const gastricCapacity = stage.medium.volume_l * stage.medium.free_solubility_um;
  const brainToNm = 1000 / p.brain_unbound_volume_l;
  const plasmaToNm = 1000 * fu / p.central_volume_l;
  const brainIn = p.bbb_in_l_h * fu / p.central_volume_l;
  const brainOut = p.bbb_out_l_h / p.brain_unbound_volume_l;

  function rates(y: Float64Array, dy: Float64Array) {
    const liquidIn = hasStomach ? stage.liquid_emptying_per_h * y[stomachLiquid] : 0;
    const solidIn = hasStomach ? stage.solid_emptying_per_h * y[stomachSolid] : 0;
    const gastricRelease = hasStomach ? transfer(y[stomachLiquid], y[stomachSolid], gastricCapacity,
      stage.formulation.dissolution_per_h, stage.formulation.precipitation_per_h) : 0;
    dy[stomachLiquid] = gastricRelease - liquidIn;
    dy[stomachSolid] = -gastricRelease - solidIn;

    // In the rapid-equilibrium scenario gutLiquid stores total intestinal parent;
    // only its soluble fraction is available for absorption.
    const soluble = Math.max(0, y[gutLiquid]);
    const absorb = p.absorption_per_h * (form ? soluble : Math.min(soluble, gutCapacity));
    const intestinalRelease = form ? transfer(y[gutLiquid], y[gutSolid], gutCapacity,
      dissolutionPerH, form.precipitation_per_h) : 0;
    dy[gutLiquid] = liquidIn + intestinalRelease - absorb - transitPerH * y[gutLiquid];
    dy[gutSolid] = solidIn - intestinalRelease - transitPerH * y[gutSolid];
    dy[transitLoss] = transitPerH * (y[gutSolid] + y[gutLiquid]);

    const totalPlasma = y[central] / p.central_volume_l;
    dy[peripheral] = p.peripheral_exchange_l_h * (totalPlasma - y[peripheral] / p.peripheral_volume_l);
    dy[brain] = brainIn * y[central] - brainOut * y[brain];
    dy[systemicLoss] = systemicCl * totalPlasma;
    dy[firstPassLoss] = (1 - available) * absorb;
    dy[central] = available * absorb - dy[peripheral] - dy[brain] - dy[systemicLoss];
    dy[brainAuc] = brainToNm * y[brain];
  }

  const y = new Float64Array(states);
  const initialDissolved = hasStomach ? stage.formulation.initial_dissolved_fraction
    : (form?.initial_dissolved_fraction ?? 1);
  y[hasStomach ? stomachLiquid : gutLiquid] = model.amountUmol * initialDissolved;
  y[hasStomach ? stomachSolid : gutSolid] = model.amountUmol * (1 - initialDissolved);

  const samples: TwinSample[] = [];
  function sample(timeH: number) {
    const instantSolid = form ? 0 : Math.max(y[gutLiquid] - gutCapacity, 0);
    samples.push({
      timeH, brainNm: y[brain] * brainToNm, plasmaNm: y[central] * plasmaToNm,
      solidUmol: y[gutSolid] + y[stomachSolid] + instantSolid,
      dissolvedUmol: y[gutLiquid] + y[stomachLiquid] - instantSolid,
    });
  }
  sample(0);
  const work = new Float64Array(states);
  const k1 = new Float64Array(states), k2 = new Float64Array(states);
  const k3 = new Float64Array(states), k4 = new Float64Array(states);
  function evaluateStage(previous: Float64Array, h: number, result: Float64Array) {
    for (let i = 0; i < states; i++) work[i] = y[i] + h * previous[i];
    rates(work, result);
  }

  let peakAmount = 0, timeToPeakH = 0, maxMassErrorUmol = 0;
  const totalSteps = Math.round(model.durationH / stepH);
  for (let n = 1; n <= totalSteps; n++) {
    rates(y, k1);
    evaluateStage(k1, stepH / 2, k2);
    evaluateStage(k2, stepH / 2, k3);
    evaluateStage(k3, stepH, k4);
    const previousBrain = y[brain];
    let mass = 0;
    for (let i = 0; i < states; i++) {
      y[i] += stepH / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      if (i !== brainAuc) mass += y[i];
      if (!Number.isFinite(y[i]) || y[i] < -1e-10) throw new Error('Invalid compartment inventory');
    }
    maxMassErrorUmol = Math.max(maxMassErrorUmol, Math.abs(mass - model.amountUmol));
    const brainFlux = brainIn * y[central] - brainOut * y[brain];
    if (k1[brain] > 0 && brainFlux < 0) {
      const peak = stepPeak(previousBrain, y[brain], k1[brain], brainFlux);
      if (peak.amount > peakAmount) {
        peakAmount = peak.amount;
        timeToPeakH = (n - 1) * stepH + peak.offset;
      }
    }
    if (y[brain] > peakAmount) { peakAmount = y[brain]; timeToPeakH = n * stepH; }
    if (n % reportEvery === 0) sample(n * stepH);
  }
  if (maxMassErrorUmol > 1e-7 * model.amountUmol) throw new Error('Parent mass balance failed');
  return {
    samples, peakBrainNm: peakAmount * brainToNm, timeToPeakH,
    brainAucNmH: y[brainAuc], transitLossUmol: y[transitLoss], maxMassErrorUmol,
    assumptions: {
      amountUmol: model.amountUmol, durationH: model.durationH,
      compound: { ...compound }, assaySource: model.assaySource,
      disposition: { ...p, gut_transit_per_h: transitPerH,
        liver_unbound_clint_l_h: clearanceLPerH, fraction_unbound_plasma: fu },
      intestinalMedium: { ...model.intestinalMedium, free_solubility_um: compound.solubilityUm },
      intestinalFormulation: form ? { ...form, dissolution_per_h: dissolutionPerH } : null,
      gastricStage: hasStomach ? stage : null, gastricSource: hasStomach ? model.gastricSource : null,
      interpretation: 'Single normalized oral input. Solubility and plasma binding are structure estimates. Human disposition and release rates are scenario assumptions. Clinical efficacy and safety are undetermined.',
    },
  };
}
