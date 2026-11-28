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
  await expect(page.locator('[data-connection-id="e2"] code')).toHaveText(['pocket_graph', 'pocket_graph']);
  await page.getByRole('button', { name: 'Back to overview' }).press('Space');
  await expect(encoder).toBeFocused();
  await page.getByRole('button', { name: 'Open Decoder', exact: true }).click();
  const rows = page.getByRole('region', { name: 'Connections' });
  await expect(rows.locator('[data-connection-id="e7"] code')).toHaveText(['molecules', 'molecules']);
  await expect(rows.locator('[data-connection-id="e8"] code')).toHaveText(['pmi_scores', 'rewards']);
  await rows.locator('[data-connection-id="e8"]').getByRole('button').last().click();
  await expect(page.locator('[data-block-id="n_rl"]')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back to overview' }).click();
  await page.getByRole('button', { name: 'Open SMILES', exact: true }).click();
  await expect(page.getByRole('button', { name: /aligned_mols/ })).toBeVisible();
  await page.getByRole('button', { name: 'Run preview' }).click();
  await expect(page.getByRole('heading', { name: /preview.*example molecules/i })).toBeVisible();
  await expect(page.locator('.stage-products li code').first()).toHaveText(/\S+/);
});

test('keeps cards, parallel endpoint pairs and connectors separate at each screen size', async ({ page }) => {
  await openEditor(page);
  for (const width of [1440, 768, 375, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await checkBounds(page.locator('.stage-flow > *'));
    await page.getByRole('button', { name: 'Open Decoder', exact: true }).click();
    await expect(page.locator('[data-block-id]')).toHaveCount(3);
    await checkBounds(page.locator('.stage-blocks > *'));
    await page.getByText('Boundary connections', { exact: true }).click();
    const rows = page.locator('[data-connection-id]:visible');
    await expect(rows).toHaveCount(8);
    await checkBounds(rows);
    await checkBounds(rows.locator(':scope > *'));
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width);
    await page.getByRole('button', { name: 'Back to overview' }).click();
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
    for (const view of ['overview', 'encoder']) {
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
  await page.getByText('Boundary connections', { exact: true }).click();
  const edge = page.locator('[data-connection-id="extra_readout"]');
  await expect(edge.locator('code')).toHaveText(['molecules', 'candidates']);
  await edge.getByRole('button', { name: /Custom readout/ }).click();
  await expect(page.getByRole('heading', { name: 'Other blocks' })).toBeVisible();
  await expect(page.locator('[data-block-id="custom"]')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Wiring canvas' }).click();
  expect(await exportGraph(page)).toEqual(graph);
});
