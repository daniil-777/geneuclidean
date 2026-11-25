import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoFiguresDir = path.resolve(__dirname, '../results/figures');

/**
 * Serves ../results/figures/ during dev/preview and copies 300-DPI PNGs into dist/figures/
 * on production builds so Git never stores duplicated binary copies.
 */
function repoFiguresPlugin(): Plugin {
  let outDir = path.resolve(__dirname, 'dist');
  let baseUrl = '/geneuclidean/';

  const createMiddleware = () => {
    return (
      req: { url?: string },
      res: {
        setHeader: (k: string, v: string) => void;
        end: (buf: Buffer) => void;
      },
      next: () => void,
    ) => {
      const rawUrl = (req.url || '').split('?')[0];
      const prefixes = [
        `${baseUrl.replace(/\/$/, '')}/figures/`,
        '/figures/',
      ];
      const matchedPrefix = prefixes.find((p) => rawUrl.startsWith(p));
      if (!matchedPrefix) {
        next();
        return;
      }
      const filename = path.basename(decodeURIComponent(rawUrl.slice(matchedPrefix.length)));
      const candidate = path.join(repoFiguresDir, filename);
      if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
        res.setHeader('Content-Type', 'image/png');
        res.setHeader('Cache-Control', 'public, max-age=3600');
        res.end(fs.readFileSync(candidate));
        return;
      }
      next();
    };
  };

  return {
    name: 'geneuclidean-repo-figures',
    configResolved(config) {
      outDir = path.resolve(config.root, config.build.outDir);
      baseUrl = config.base || '/';
    },
    configureServer(server) {
      server.middlewares.use(createMiddleware());
    },
    configurePreviewServer(server) {
      server.middlewares.use(createMiddleware());
    },
    closeBundle() {
      if (!fs.existsSync(repoFiguresDir)) return;
      const targetDir = path.join(outDir, 'figures');
      fs.mkdirSync(targetDir, { recursive: true });
      for (const entry of fs.readdirSync(repoFiguresDir)) {
        if (entry.endsWith('.png')) {
          fs.copyFileSync(path.join(repoFiguresDir, entry), path.join(targetDir, entry));
        }
      }
    },
  };
}

export default defineConfig({
  base: '/geneuclidean/',
  plugins: [react(), tailwindcss(), repoFiguresPlugin()],
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1800,
    rollupOptions: {
      output: {
        manualChunks: {
          charts: ['recharts'],
          icons: ['lucide-react'],
        },
      },
    },
  },
});
