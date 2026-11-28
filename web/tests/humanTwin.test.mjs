import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { defaultTwinInputs, leadProperties, simulateHumanTwin, twinRanges } from '../src/lib/humanTwin.ts';
import reference from './humanTwinReference.json' with { type: 'json' };

function close(actual, expected, tolerance, label) {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, calculated ${actual} (tolerance ${tolerance})`);
}

// These cases are generated independently by SciPy using both real study
// configurations and a simultaneous perturbation of the three interactive rates.
for (const fixture of reference.cases) {
  test(`Python parity, ${fixture.lead}, ${fixture.formulation}${fixture.custom ? ', custom rates' : ''}`, () => {
    const inputs = defaultTwinInputs(fixture.lead, fixture.formulation);
    for (const key of ['transitPerH', 'clearanceLPerH', 'dissolutionPerH']) {
      if (fixture.custom) inputs[key] = fixture[key];
      else assert.equal(inputs[key], fixture[key], key);
    }
    const run = simulateHumanTwin(inputs);
    close(run.peakBrainNm, fixture.peakBrainNm, 2e-7, 'trajectory peak (nM)');
    close(run.timeToPeakH, fixture.timeToPeakH, 3e-5, 'time to peak (h)');
    close(run.brainAucNmH, fixture.brainAucNmH, 1e-6, 'AUC (nM h)');
    close(run.transitLossUmol, fixture.transitLossUmol, 1e-8, 'unabsorbed transit (µmol)');
    assert.ok(run.maxMassErrorUmol < 1e-12);
    assert.equal(run.samples.length, 97);
    for (const [i, hour] of reference.timesH.entries()) {
      const sample = run.samples[hour * 2];
      assert.equal(sample.timeH, hour);
      for (const [field, tolerance] of Object.entries({
        brainNm: 4e-7, plasmaNm: 6e-6, solidUmol: 2e-7, dissolvedUmol: 2e-7,
      })) close(sample[field], fixture[field][i], tolerance, `${field} at ${hour} h`);
    }
  });
}

test('all half-hourly samples agree with the published flagship report', () => {
  const report = JSON.parse(readFileSync(
    new URL('../../results/tables/tourette_human_exposure.json', import.meta.url), 'utf8'));
  for (const [name, ref] of Object.entries(report.formulation_experiment.cases)) {
    const run = simulateHumanTwin(defaultTwinInputs(report.compound.compound_id, name));
    for (const [i, sample] of run.samples.entries()) {
      close(sample.brainNm, ref.unbound_brain_nm[i], 5e-7, 'brain nM');
      close(sample.plasmaNm, ref.unbound_plasma_nm[i], 6e-6, 'plasma nM');
      close(sample.solidUmol, ref.gut_solid_umol[i], 2e-7, 'solid µmol');
      close(sample.dissolvedUmol, ref.gut_dissolved_umol[i], 2e-7, 'dissolved µmol');
    }
  }
});

test('both compounds conserve parent mass at every slider boundary', () => {
  const forms = [...new Set(reference.cases.map(row => row.formulation))];
  for (const lead of Object.keys(leadProperties)) {
    for (const formulation of forms) {
      for (const transitPerH of Object.values(twinRanges.transitPerH)) {
        for (const clearanceLPerH of Object.values(twinRanges.clearanceLPerH)) {
          for (const dissolutionPerH of Object.values(twinRanges.dissolutionPerH)) {
            const run = simulateHumanTwin({ lead, formulation, transitPerH, clearanceLPerH, dissolutionPerH });
            assert.ok(run.maxMassErrorUmol < 1e-12);
            assert.ok(run.peakBrainNm > 0 && run.timeToPeakH > 0 && run.timeToPeakH < 48);
            assert.ok(run.transitLossUmol >= 0 && run.transitLossUmol <= 1);
            assert.ok(run.samples.every(s => Object.values(s).every(v => Number.isFinite(v) && v >= 0)));
          }
        }
      }
    }
  }
});

test('malformed rates and unknown presets cannot enter the solver', () => {
  const base = defaultTwinInputs('DRD1_1', 'rapid_equilibrium');
  for (const [key, bounds] of Object.entries(twinRanges)) {
    for (const value of [null, NaN, Infinity, bounds.min - 0.001, bounds.max + 0.001]) {
      assert.throws(() => simulateHumanTwin({ ...base, [key]: value }), RangeError);
    }
  }
  for (const invalid of ['other', '__proto__', 'constructor']) {
    assert.throws(() => simulateHumanTwin({ ...base, lead: invalid }), RangeError);
    assert.throws(() => defaultTwinInputs(base.lead, invalid), RangeError);
  }
});
