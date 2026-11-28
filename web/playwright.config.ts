import { defineConfig } from '@playwright/test';

const baseURL = process.env.STUDIO_URL ?? 'http://127.0.0.1:4188/geneuclidean/';
export default defineConfig({
  testDir: './tests/browser',
  workers: 2,
  use: { baseURL, viewport: { width: 1440, height: 960 }, reducedMotion: 'reduce', trace: 'retain-on-failure' },
  webServer: process.env.STUDIO_URL ? undefined : {
    command: 'npm run preview -- --host 127.0.0.1 --port 4188 --strictPort',
    url: baseURL,
    reuseExistingServer: !process.env.CI,
  },
});
