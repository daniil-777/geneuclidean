import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import type { CircuitGraphV1 } from '../../src/types/circuit';

async function openEditor(page: Page) {
  await page.goto('./#circuit');
  await expect(page.getByRole('heading', { name: 'Circuit editor' })).toBeVisible();
}

async function exportGraph(page: Page): Promise<CircuitGraphV1> {
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON', exact: true }).click();
  return JSON.parse(await readFile((await (await downloaded).path())!, 'utf8'));
}

// Check actual rendered bounds, including connectors, at each breakpoint. A diagram
// can have the correct labels and still place two controls or a wire on top of them.
async function checkBounds(items: Locator) {
  const problems = await items.evaluateAll(elements => {
    const rects = elements.map(e => e.getBoundingClientRect());
    const problems: string[] = [];
    for (const [i, r] of rects.entries()) {
      if (!r.width || !r.height || r.left < -1 || r.right > window.innerWidth + 1) {
        problems.push('Outside viewport or hidden: ' + i);
      }
      for (let j = 0; j < i; j++) {
        const s = rects[j];
        const overlapX = Math.min(r.right, s.right) - Math.max(r.left, s.left);
        const overlapY = Math.min(r.bottom, s.bottom) - Math.max(r.top, s.top);
        if (overlapX > 1 && overlapY > 1) problems.push(`Overlap between ${i} and ${j}`);
      }
    }
    return problems;
  });
  expect(problems).toEqual([]);
}

// Inspect the drawn SVG in CSS pixels so the test also catches transforms, layout
// breakpoints, detached handles and connectors painted across block labels.
async function readDiagram(stage: Locator) {
  return stage.evaluate(root => {
    const nodes = [...root.querySelectorAll('[data-diagram-node]')];
    const ports = [...root.querySelectorAll<SVGCircleElement>('circle[data-port-id]')];
    const cards = nodes.map(el => el.getBoundingClientRect());
    const errors: string[] = [];
    const wires = [...root.querySelectorAll<SVGPathElement>('path[data-wire-id]')].map(wire => {
      const length = wire.getTotalLength();
      const transform = wire.getScreenCTM()!;
      const points = Array.from({ length: 65 }, (_, i) => {
        const p = wire.getPointAtLength(length * i / 64).matrixTransform(transform);
        return { x: p.x, y: p.y };
      });
      const { wireId, source, sourcePort, target, targetPort } = wire.dataset;
      for (const [index, nodeId, portId] of [[0, source, sourcePort], [64, target, targetPort]] as const) {
        const port = ports.find(p => p.dataset.nodeId === nodeId && p.dataset.portId === portId &&
          p.dataset.edgeId === wireId && p.dataset.portDirection === (index === 0 ? 'output' : 'input'));
        if (!port) { errors.push(`${wireId}: missing ${nodeId}.${portId}`); continue; }
        const r = port.getBoundingClientRect();
        const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        const node = nodes.find(n => n.getAttribute('data-node-id') === nodeId)?.getBoundingClientRect();
        if (!node || Math.min(Math.abs(cx - node.left), Math.abs(cx - node.right),
              Math.abs(cy - node.top), Math.abs(cy - node.bottom)) > 2 ||
            cx < node.left - 2 || cx > node.right + 2 || cy < node.top - 2 || cy > node.bottom + 2) {
          errors.push(`${wireId}: ${portId} floats away from its block`);
        }
        const d = Math.hypot(points[index].x - cx, points[index].y - cy);
        if (d > r.width / 2 + 1) errors.push(`${wireId}: misses ${nodeId}.${portId} by ${d.toFixed(1)}px`);
      }
      if (points.some(p => cards.some(r => p.x > r.left + 2 && p.x < r.right - 2 && p.y > r.top + 2 && p.y < r.bottom - 2))) {
        errors.push(`${wireId}: runs through a block`);
      }
      if (length < 8 || getComputedStyle(wire).stroke === 'none') errors.push(`${wireId}: invisible route`);
      return { id: wireId, source, sourcePort, target, targetPort, points };
    });
    for (const [i, a] of cards.entries()) {
      if (cards.slice(0, i).some(b => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
          Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1)) errors.push(`Overlapping card ${i}`);
    }
    return { errors, wires };
  });
}

async function openConnectionList(page: Page) {
  await page.locator('summary').filter({ hasText: /^Connection list/ }).click();
}

function mode(page: Page, name: 'Diagram' | 'Parameter form') {
  return page.getByRole('group', { name: 'Stage editor mode' }).getByRole('button', { name, exact: true });
}

test('traces the default pocket-to-SMILES path and its two parallel decoder signals by keyboard', async ({ page }) => {
  await openEditor(page);
  await expect(page.getByRole('combobox', { name: 'Preset' })).toHaveValue('tourette_scz_champion');
  const path = page.getByRole('region', { name: 'Signal path' });
  await expect(path.getByRole('button')).toHaveCount(4);
  await expect(path.locator('[data-stage-id] > strong')).toHaveText(['Pocket', 'Encoder', 'Decoder', 'SMILES']);
  await expect(path.locator('.stage-signal-arrow')).toHaveCount(3);
  await expect(page.getByRole('heading', { name: 'Key parameters' })).toHaveCount(0);

  const encoder = page.getByRole('button', { name: 'Open Encoder', exact: true });
  await encoder.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Encoder', exact: true })).toBeFocused();
  await expect(mode(page, 'Diagram')).toHaveAttribute('aria-pressed', 'true');
  const encoderWire = page.locator('path[data-wire-id="e2"]');
  await expect(encoderWire).toHaveAttribute('data-source-port', 'pocket_graph');
  await expect(encoderWire).toHaveAttribute('data-target-port', 'pocket_graph');
  await page.getByRole('button', { name: 'Back to overview' }).press('Space');
  await expect(encoder).toBeFocused();
  await page.getByRole('button', { name: 'Open Decoder', exact: true }).click();
  const reward = page.locator('path[data-wire-id="e8"]');
  await reward.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-wire-detail]')).toContainText('pmi_scores');
  await expect(page.locator('[data-wire-detail]')).toContainText('rewards');
  await openConnectionList(page);
  const rewardsRow = page.locator('[data-connection-id="e8"]');
  await expect(rewardsRow.locator('code')).toHaveText(['pmi_scores', 'rewards']);
  await rewardsRow.getByRole('button').last().press('Space');
  await expect(page.locator('[data-block-id="n_rl"]')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back to overview' }).click();
  await page.getByRole('button', { name: 'Open SMILES', exact: true }).click();
  await expect(page.getByRole('button', { name: /aligned_mols/ })).toBeVisible();
  await page.getByRole('button', { name: 'Run preview' }).click();
  await expect(page.getByRole('heading', { name: /preview.*example molecules/i })).toBeVisible();
  await expect(page.locator('.stage-products li code').first()).toHaveText(/\S+/);
});

test('attaches each wire to its declared ports and separates both parallel decoder paths at each width', async ({ page }) => {
  await openEditor(page);
  for (const width of [1440, 768, 375, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await checkBounds(page.locator('.stage-flow > *'));
    for (const stage of ['Encoder', 'Decoder']) {
      await page.getByRole('button', { name: 'Open ' + stage, exact: true }).click();
      const diagram = page.locator('[data-diagram-stage]');
      await expect(diagram.locator('[data-diagram-node]')).toHaveCount(stage === 'Encoder' ? 2 : 3);
      const { errors, wires } = await readDiagram(diagram);
      expect(errors, `${stage} at ${width}px`).toEqual([]);
      expect(wires.map(w => [w.id, w.source, w.sourcePort, w.target, w.targetPort])).toEqual(stage === 'Encoder'
        ? [['e2', 'n_cart', 'pocket_graph', 'n_bridge', 'pocket_graph']]
        : [['e6', 'n_lm', 'molecules', 'n_cfg', 'lm_candidates'],
           ['e7', 'n_cfg', 'molecules', 'n_rl', 'molecules'], ['e8', 'n_cfg', 'pmi_scores', 'n_rl', 'rewards']]);
      const positions = await diagram.locator('[data-diagram-node]').evaluateAll(els => els.map(el => {
        const r = el.getBoundingClientRect();
        return { x: r.left, y: r.top };
      }));
      if (width === 1440) expect(positions[1].x).toBeGreaterThan(positions[0].x + 10);
      if (width <= 375) expect(positions[1].y).toBeGreaterThan(positions[0].y + 10);
      if (stage === 'Decoder') {
        const a = wires.find(w => w.id === 'e7')!.points;
        const b = wires.find(w => w.id === 'e8')!.points;
        expect(Math.min(...a.flatMap(p => b.map(q => Math.hypot(p.x - q.x, p.y - q.y))))).toBeGreaterThan(2);
        if (width === 1440) {
          const zoom = diagram.getByLabel('Zoom level');
          const fit = await zoom.textContent();
          await diagram.getByRole('button', { name: 'Zoom in', exact: true }).click();
          await expect(zoom).not.toHaveText(fit!);
          expect((await readDiagram(diagram)).errors).toEqual([]);
          await diagram.getByRole('button', { name: 'Fit diagram' }).click();
          await expect(zoom).toHaveText(fit!);
          const wire = diagram.locator('path[data-wire-id="e7"]');
          await wire.scrollIntoViewIfNeeded();
          const mid = await wire.evaluate(el => {
            const path = el as SVGPathElement;
            const p = path.getPointAtLength(path.getTotalLength() / 2).matrixTransform(path.getScreenCTM()!);
            return { x: p.x, y: p.y };
          });
          await page.mouse.click(mid.x, mid.y);
          await expect(wire).toHaveAttribute('aria-pressed', 'true');
        }
        await openConnectionList(page);
        await page.getByText('Boundary connections', { exact: true }).click();
        const rows = page.locator('[data-connection-id]:visible');
        await expect(rows).toHaveCount(8);
        await checkBounds(rows);
        await checkBounds(rows.locator(':scope > *'));
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
      await page.getByRole('button', { name: 'Back to overview' }).click();
    }
  }
});

test('an edit survives switching views and exports only the chosen parameter change', async ({ page }) => {
  await openEditor(page);
  await page.getByRole('button', { name: 'Wiring canvas' }).click();
  const original = await exportGraph(page);
  await page.getByRole('button', { name: 'Stage view' }).click();
  await page.getByRole('button', { name: 'Open Encoder', exact: true }).click();
  await page.getByRole('button', { name: 'Inspect CartFormer-SE3', exact: true }).click();
  const layers = page.getByRole('slider', { name: 'Layers', exact: true });
  await expect(layers).toHaveValue('4');
  await layers.press('ArrowRight');
  await expect(layers).toHaveValue('5');
  await page.getByRole('button', { name: 'Undo edit' }).click();
  await expect(layers).toHaveValue('4');
  await page.getByRole('button', { name: 'Redo edit' }).click();
  await expect(layers).toHaveValue('5');
  await page.getByRole('button', { name: 'Edit in full canvas' }).click();
  await expect(page.getByRole('button', { name: 'Wiring canvas' })).toHaveAttribute('aria-pressed', 'true');
  const expected = structuredClone(original);
  expected.nodes.find(n => n.id === 'n_cart')!.params.num_layers = 5;
  expect(await exportGraph(page)).toEqual(expected);
});

test('preserves feedback, offers valid empty-state actions and clears drilldown on preset change', async ({ page }) => {
  await openEditor(page);
  const preset = page.getByRole('combobox', { name: 'Preset' });
  await page.getByRole('button', { name: 'Open Decoder', exact: true }).click();
  await preset.selectOption('al_dmta_closed_loop');
  await expect(page.getByRole('heading', { name: 'Signal path' })).toBeVisible();
  await page.getByRole('button', { name: 'Open Active learning' }).click();
  await openConnectionList(page);
  await page.getByText('Boundary connections', { exact: true }).click();
  const feedback = page.locator('[data-connection-id="e_fb_1"]');
  await expect(feedback).toContainText('Feedback');
  await expect(feedback.locator('code')).toHaveText(['scores', 'observations']);
  await preset.selectOption('nanofold3_cofold_audit');
  await expect(page.getByRole('button', { name: 'Open SMILES' })).toHaveCount(0);
  await preset.selectOption('blank_breadboard');
  await expect(page.getByRole('heading', { name: 'This circuit is empty' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run preview' })).toBeDisabled();
  await page.getByRole('button', { name: 'Open full canvas' }).click();
  await expect(page.getByRole('button', { name: 'Wiring canvas' })).toHaveAttribute('aria-pressed', 'true');
});

for (const theme of ['light', 'dark']) {
  test(`stage overview and encoder settings pass automated accessibility checks in ${theme} mode`, async ({ page }) => {
    await openEditor(page);
    if (theme === 'dark') await page.getByRole('button', { name: 'Switch to dark mode' }).click();
    for (const view of ['overview', 'encoder', 'form']) {
      if (view === 'form') await mode(page, 'Parameter form').click();
      if (view === 'encoder') {
        await page.getByRole('button', { name: 'Open Encoder', exact: true }).click();
        await page.getByRole('button', { name: 'Inspect CartFormer-SE3', exact: true }).click();
      }
      const { violations } = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      expect(violations.map(v => ({ rule: v.id, elements: v.nodes.map(n => n.target) }))).toEqual([]);
    }
  });
}

test('keeps the same preview after auto-layout and clears it after an encoder edit', async ({ page }) => {
  await openEditor(page);
  const smiles = page.getByRole('button', { name: 'Open SMILES', exact: true });
  const examples = page.locator('.stage-products li code');
  await smiles.click();
  await page.getByRole('button', { name: 'Run preview' }).click();
  await expect(examples.first()).toHaveText(/\S+/);
  const initialExamples = await examples.allTextContents();

  await page.getByRole('button', { name: 'Wiring canvas' }).click();
  const original = await exportGraph(page);
  await page.getByRole('button', { name: 'Auto-Layout', exact: true }).click();
  const laidOut = await exportGraph(page);
  expect(laidOut.nodes.map(n => n.position)).not.toEqual(original.nodes.map(n => n.position));
  await page.getByRole('button', { name: 'Stage view' }).click();
  await expect(page.getByRole('heading', { name: 'SMILES', exact: true })).toBeVisible();
  await expect(examples).toHaveText(initialExamples);
  await expect(page.getByRole('heading', { name: /preview.*example molecules/i })).toBeVisible();

  await page.getByRole('button', { name: 'Back to overview' }).click();
  await page.getByRole('button', { name: 'Open Encoder', exact: true }).click();
  await page.getByRole('button', { name: 'Inspect CartFormer-SE3', exact: true }).click();
  await page.getByRole('slider', { name: 'Layers', exact: true }).press('ArrowRight');
  await page.getByRole('button', { name: 'Back to overview' }).click();
  await smiles.click();
  await expect(examples).toHaveCount(0);
  await expect(page.getByText('Run preview to inspect reference SMILES.')).toBeVisible();
});

test('imports distinct decoder outputs and preserves an unknown consumer in the graph', async ({ page }) => {
  await openEditor(page);
  await page.getByRole('button', { name: 'Wiring canvas' }).click();
  const graph = await exportGraph(page);
  graph.id = 'parallel-decoder-import';
  graph.name = 'Two decoder outputs';
  delete graph.metadata?.presetId;
  graph.nodes.push(
    { id: 'second', blockType: 'safe_gpt', label: 'Additional decoder', position: { x: 80, y: 860 }, params: {} },
    { id: 'custom', blockType: 'custom_readout', label: 'Custom readout', position: { x: 380, y: 860 }, params: { revision: 2 } },
  );
  graph.edges.push(
    { id: 'extra_graph', source: 'n_cart', sourcePort: 'pocket_graph', target: 'second', targetPort: 'pocket_graph', signalType: 'SE3_GRAPH', kind: 'forward' },
    { id: 'extra_readout', source: 'second', sourcePort: 'molecules', target: 'custom', targetPort: 'candidates', signalType: 'MOL_BATCH', kind: 'forward' },
  );
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Import JSON' }).click();
  await (await chooser).setFiles({ name: 'parallel-decoders.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(graph)) });
  await page.getByRole('button', { name: 'Stage view' }).click();
  const preset = page.getByRole('combobox', { name: 'Preset' });
  await expect(preset.locator('option:checked')).toContainText(graph.name);
  await page.getByRole('button', { name: 'Open SMILES', exact: true }).click();
  const products = page.locator('.stage-products button');
  await expect(products).toHaveCount(2);
  await expect(products.filter({ hasText: 'aligned_mols' })).toBeVisible();
  await products.filter({ hasText: 'Additional decoder' }).click();
  await expect(page.locator('[data-block-id="second"]')).toHaveAttribute('aria-pressed', 'true');
  await openConnectionList(page);
  await page.getByText('Boundary connections', { exact: true }).click();
  const edge = page.locator('[data-connection-id="extra_readout"]');
  await expect(edge.locator('code')).toHaveText(['molecules', 'candidates']);
  await edge.getByRole('button', { name: /Custom readout/ }).click();
  await expect(page.getByRole('heading', { name: 'Other blocks' })).toBeVisible();
  await expect(page.locator('[data-block-id="custom"]')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Wiring canvas' }).click();
  expect(await exportGraph(page)).toEqual(graph);
});


test('edits advanced settings by keyboard in the form, preserving selection, value and mode', async ({ page }) => {
  await openEditor(page);
  await page.getByRole('button', { name: 'Open Encoder', exact: true }).press('Enter');
  await mode(page, 'Parameter form').press('Space');
  await expect(mode(page, 'Parameter form')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('[data-diagram-stage]')).toHaveCount(0);
  const block = page.getByRole('combobox', { name: 'Block', exact: true });
  await block.press('p');
  await expect(block).toHaveValue('n_bridge');
  const advanced = page.getByRole('slider', { name: 'ScaleMatch RMS', exact: true });
  await expect(advanced).toBeHidden();
  await page.locator('summary').filter({ hasText: /^More parameters/ }).press('Enter');
  await expect(advanced).toHaveValue('1');
  await advanced.press('ArrowRight');
  await expect(advanced).toHaveValue('1.1');
  await mode(page, 'Diagram').press('Enter');
  await expect(page.locator('[data-block-id="n_bridge"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(advanced).toBeHidden();
  await mode(page, 'Parameter form').press('Enter');
  await expect(block).toHaveValue('n_bridge');
  await page.getByRole('button', { name: 'Edit in full canvas' }).press('Enter');
  expect((await exportGraph(page)).nodes.find(n => n.id === 'n_bridge')?.params.output_rms_scale).toBe(1.1);
  await page.getByRole('button', { name: 'Stage view' }).click();
  await expect(mode(page, 'Parameter form')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back to overview' }).click();
  await page.getByRole('button', { name: 'Open Decoder', exact: true }).click();
  await expect(mode(page, 'Parameter form')).toHaveAttribute('aria-pressed', 'true');
  await page.reload();
  await page.getByRole('button', { name: 'Open Encoder', exact: true }).click();
  await expect(mode(page, 'Parameter form')).toHaveAttribute('aria-pressed', 'true');
});

test('links single-block diagrams to the implementation and keeps code steps separate from circuit wires', async ({ page }) => {
  await openEditor(page);
  await page.getByRole('combobox', { name: 'Preset' }).selectOption('se3_denovo_cns');
  const operations = {
    Encoder: ['build_radius_knn_graph(', 'self.node_embed(', 'self.message_blocks', 'self.out_proj('],
    Decoder: ['self.perceiver(', 'select_next_tokens(', 'safe_to_smiles('],
  };
  for (const [name, signatures] of Object.entries(operations)) {
    await page.getByRole('button', { name: 'Open ' + name, exact: true }).click();
    const diagram = page.locator('[data-diagram-stage]');
    await expect(diagram.locator('[data-diagram-node]')).toHaveCount(1);
    await expect(diagram.locator('path[data-wire-id]')).toHaveCount(0);
    const steps = diagram.locator('[data-implementation-step]');
    await expect(steps).toHaveCount(signatures.length);
    for (const [i, signature] of signatures.entries()) {
      const url = new URL((await steps.nth(i).getAttribute('href'))!);
      expect(url.origin).toBe('https://github.com');
      expect(url.pathname).toMatch(/^\/daniil-777\/geneuclidean\/blob\/[^/]+\/src\/geneuclidean\/.*\.py$/);
      expect(url.hash).toMatch(/^#L\d+$/);
      const filename = url.pathname.slice(url.pathname.indexOf('src/geneuclidean/'));
      const source = await readFile(new URL('../../../' + filename, import.meta.url), 'utf8');
      expect(source.split('\n')[Number(url.hash.slice(2)) - 1]).toContain(signature);
    }
    await page.getByRole('button', { name: 'Back to overview' }).click();
  }
});
