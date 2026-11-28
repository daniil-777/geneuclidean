import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('opens the simulator, recomputes exposure and reopens an exported scenario', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('./');
  await page.getByRole('link', { name: 'Open simulator', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Tourette simulator' })).toBeVisible();
  const metrics = page.getByRole('region', { name: 'Modeled unbound brain exposure' });
  await expect(metrics).toContainText('3.36');
  await page.getByRole('button', { name: 'Slow release', exact: true }).click();
  await expect(metrics).toContainText('1.63');
  await expect(metrics).toContainText('6.24');
  await page.getByRole('combobox', { name: 'Research compound' }).selectOption('DRD1_1');
  const clearance = page.getByRole('slider', { name: 'Liver intrinsic clearance' });
  await clearance.press('ArrowRight');
  await expect(clearance).toHaveValue('11');
  await page.getByRole('button', { name: 'Plasma', exact: true }).click();
  await expect(page.getByRole('img', { name: /unbound plasma concentration/i })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download JSON' }).click();
  const file = await download;
  const report = JSON.parse(await readFile((await file.path())!, 'utf8'));
  expect(report.inputs).toMatchObject({ lead: 'DRD1_1', formulation: 'slower_solid_release', clearanceLPerH: 11 });
  expect(report.exposure.samples).toHaveLength(97);
  expect(report.exposure.maxMassErrorUmol).toBeLessThan(1e-7);
  await page.getByRole('button', { name: 'Reset parameters' }).click();
  await expect(clearance).toHaveValue('10');
  await page.goto(report.scenarioUrl);
  await page.reload();
  await expect(clearance).toHaveValue('11');
  await expect(page.getByRole('combobox', { name: 'Research compound' })).toHaveValue('DRD1_1');
  await expect(metrics).toContainText(report.exposure.peakBrainNm.toFixed(2));
  expect(errors).toEqual([]);
});

test('opens on a phone, preserves theme and supports browser history', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto('./#human-twin');
  await expect(page.getByRole('heading', { name: 'Tourette simulator' })).toBeVisible();
  await page.getByRole('slider', { name: 'Liver intrinsic clearance' }).press('ArrowRight');
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Switch to light mode' })).toBeVisible();
  await expect(page.getByRole('slider', { name: 'Liver intrinsic clearance' })).toHaveValue('11');
  await page.getByRole('button', { name: 'Open navigation' }).click();
  const nav = page.getByRole('navigation', { name: 'Studio navigation' });
  await nav.getByRole('link', { name: 'Overview', exact: true }).click();
  await expect(page.getByRole('link', { name: 'Open simulator', exact: true })).toBeVisible();
  await page.goBack();
  await expect(page.getByRole('heading', { name: 'Tourette simulator' })).toBeVisible();
  await expect(page.getByRole('slider', { name: 'Liver intrinsic clearance' })).toHaveValue('11');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
});

for (const tab of ['overview', 'human-twin']) {
  test(tab + ' passes automated accessibility checks', async ({ page }) => {
    await page.goto('./#' + tab);
    await expect(page.locator('main h1')).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    for (const theme of ['light', 'dark']) {
      if (theme === 'dark') await page.getByRole('button', { name: 'Switch to dark mode' }).click();
      const { violations } = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      expect(violations.map(v => ({ id: v.id, elements: v.nodes.map(n => n.target) }))).toEqual([]);
    }
  });
}

test('recovers a failed dataset request and opens the twin independently', async ({ page }) => {
  await page.route('**/data/overview.json', route => route.abort(), { times: 1 });
  await page.goto('./');
  await expect(page.getByRole('alert')).toContainText('dataset');
  await page.getByRole('button', { name: 'Try again' }).click();
  await expect(page.getByRole('link', { name: 'Open simulator', exact: true })).toBeVisible();
  await page.route('**/data/*.json', route => route.abort());
  await page.goto('./#human-twin');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Download JSON' })).toBeVisible();
});

test('chart covers 48 hours at each width and sharing works with clipboard disabled', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', { value: undefined }));
  await page.goto('./#human-twin?lead=DRD1_1&form=slower_solid_release&cl=20');
  await expect(page.getByRole('slider', { name: 'Liver intrinsic clearance' })).toHaveValue('20');
  for (const width of [1440, 768, 375]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(() => page.evaluate(() => {
      const svg = document.querySelector<SVGSVGElement>('.ht-plot svg');
      const curve = svg?.querySelector<SVGPathElement>('.ht-active-line');
      const grid = svg?.querySelector<SVGLineElement>('.ht-gridline');
      const line = svg?.querySelector<SVGLineElement>('.ht-cursor');
      const dot = svg?.querySelector<SVGCircleElement>('.ht-cursor-point');
      if (!svg || !curve || !grid || !line || !dot) return false;
      const end = curve.getPointAtLength(curve.getTotalLength());
      return Math.abs(end.x - grid.x2.baseVal.value) < 0.02 &&
        Math.abs(dot.cx.baseVal.value - line.x1.baseVal.value) < 0.02 &&
        document.documentElement.scrollWidth <= window.innerWidth;
    })).toBe(true);
  }
  await page.getByRole('button', { name: 'Copy link' }).click();
  await expect(page.getByRole('textbox', { name: 'Copy this scenario link' })).toHaveValue(/lead=DRD1_1.*cl=20/);
});
