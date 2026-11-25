import { assetUrl } from './data';

export interface GeneratorConfig {
  vocab_size: number;
  pad_id: number;
  bos_id: number;
  eos_id: number;
  unk_id: number;
  d_model: number;
  n_heads: number;
  head_dim: number;
  n_layers: number;
  d_ff: number;
  pocket_dim: number;
  cond_dim: number;
  film_dim: number;
  prefix_len: number;
  max_seq_len: number;
  rope_base: number;
  rms_eps: number;
  tie_word_embeddings: boolean;
}

export interface TensorMeta {
  dtype: 'float32' | 'int8_row';
  shape: number[];
  byte_offset: number;
  byte_length: number;
  scale_offset?: number;
  scale_length?: number;
}

export interface GeneratorManifest {
  schema_version: string;
  weights_file: string;
  weights_sha256?: string;
  weights_bytes: number;
  config: GeneratorConfig;
  vocab: string[];
  defaults: {
    temperature: number;
    top_p: number;
    top_k: number;
    cfg_scale: number;
    max_new_tokens: number;
  };
  tensors: Record<string, TensorMeta>;
  verification?: Record<string, unknown>;
}

export interface PocketTargetEmbedding {
  target_id: string;
  gene_symbol: string;
  name: string;
  target_class: string;
  uniprot_id: string;
  chembl_id: string;
  primary_pdb_id: string;
  indications: string[];
  therapeutic_moa: string;
  desired_state: string;
  primary_anchor_residue: string;
  hotspot_residues: string[];
  reference_drugs: string[];
  embedding: number[];
}

export interface GeneratorPreset {
  id: string;
  label: string;
  description: string;
  mode: 'single' | 'contrastive' | 'dual_blend';
  primary_target: string;
  anti_target?: string;
  secondary_target?: string;
  alpha?: number;
  blend_t?: number;
  cfg_scale: number;
  temperature: number;
}

export interface PocketEmbeddingsPack {
  schema_version: string;
  encoder: string;
  dim: number;
  normalization: string;
  center_mean: number[];
  null_embedding: number[];
  targets: Record<string, PocketTargetEmbedding>;
  presets: GeneratorPreset[];
}

interface Int8Matrix {
  q: Int8Array;
  scales: Float32Array;
  rows: number;
  cols: number;
}

interface LayerWeights {
  norm1_w: Float32Array;
  norm2_w: Float32Array;
  film_w: Int8Matrix;
  film_b: Float32Array;
  q_w: Int8Matrix;
  k_w: Int8Matrix;
  v_w: Int8Matrix;
  o_w: Int8Matrix;
  w1_w: Int8Matrix;
  w3_w: Int8Matrix;
  w2_w: Int8Matrix;
}

export interface GenerateOptions {
  zCond: Float32Array;
  cfgScale: number;
  temperature: number;
  topK: number;
  topP: number;
  maxNewTokens: number;
  seed?: number;
  onToken?: (partialSmiles: string) => void;
}

export interface GeneratedSample {
  smiles: string;
  tokenCount: number;
  latencyMs: number;
}

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

function rmsNorm(out: Float32Array, x: Float32Array, w: Float32Array, eps: number): void {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
}

function matvecInt8(
  out: Float32Array,
  mat: Int8Matrix,
  x: Float32Array,
  bias?: Float32Array,
): void {
  const { q, scales, rows, cols } = mat;
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) {
      acc += q[off + c] * x[c];
    }
    out[r] = acc * scales[r] + (bias ? bias[r] : 0);
  }
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}

export function buildSteeringVector(
  pack: PocketEmbeddingsPack,
  mode: 'single' | 'contrastive' | 'dual_blend',
  primaryId: string,
  secondaryId?: string,
  param = 0.5,
): Float32Array {
  const dim = pack.dim;
  const zA = pack.targets[primaryId]?.embedding;
  const out = new Float32Array(dim);
  if (!zA) return out;

  if (mode === 'single' || !secondaryId || !pack.targets[secondaryId]) {
    for (let i = 0; i < dim; i++) out[i] = zA[i];
    return out;
  }
  const zB = pack.targets[secondaryId].embedding;
  let normSq = 0;
  for (let i = 0; i < dim; i++) {
    const v =
      mode === 'contrastive' ? zA[i] - param * zB[i] : (1 - param) * zA[i] + param * zB[i];
    out[i] = v;
    normSq += v * v;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 1e-8) {
    for (let i = 0; i < dim; i++) out[i] /= norm;
  }
  return out;
}

/**
 * Pure-TypeScript Per-Row INT8 KV-Cached Causal Chemical Transformer with 2-Stream CFG.
 * Zero WASM dependencies; exact `< 1e-4` numerical parity with Python INT8 dequantization.
 */
export class BrowserChemicalTransformer {
  readonly manifest: GeneratorManifest;
  readonly cfg: GeneratorConfig;

  private tokEmb: Float32Array;
  private lmHeadBias: Float32Array;
  private nullPrefix: Float32Array;
  private pocketFc1W: Int8Matrix;
  private pocketFc1B: Float32Array;
  private pocketFc2W: Int8Matrix;
  private pocketFc2B: Float32Array;
  private prefixProjW: Int8Matrix;
  private prefixProjB: Float32Array;
  private prefixNormW: Float32Array;
  private filmFcW: Int8Matrix;
  private filmFcB: Float32Array;
  private layers: LayerWeights[];
  private finalNormW: Float32Array;
  private ropeCos: Float32Array;
  private ropeSin: Float32Array;

  constructor(manifest: GeneratorManifest, buffer: ArrayBuffer) {
    this.manifest = manifest;
    this.cfg = manifest.config;

    const getF32 = (name: string): Float32Array => {
      const m = manifest.tensors[name];
      if (!m) throw new Error(`Missing float32 tensor: ${name}`);
      return new Float32Array(buffer, m.byte_offset, m.byte_length / 4);
    };

    const getI8 = (name: string): Int8Matrix => {
      const m = manifest.tensors[name];
      if (!m || m.scale_offset === undefined || m.scale_length === undefined) {
        throw new Error(`Missing int8_row tensor: ${name}`);
      }
      return {
        q: new Int8Array(buffer, m.byte_offset, m.byte_length),
        scales: new Float32Array(buffer, m.scale_offset, m.scale_length / 4),
        rows: m.shape[0],
        cols: m.shape[1],
      };
    };

    this.tokEmb = getF32('tok_emb');
    this.lmHeadBias = getF32('lm_head_bias');
    this.nullPrefix = getF32('null_prefix');
    this.pocketFc1W = getI8('pocket_fc1_w');
    this.pocketFc1B = getF32('pocket_fc1_b');
    this.pocketFc2W = getI8('pocket_fc2_w');
    this.pocketFc2B = getF32('pocket_fc2_b');
    this.prefixProjW = getI8('prefix_proj_w');
    this.prefixProjB = getF32('prefix_proj_b');
    this.prefixNormW = getF32('prefix_norm_w');
    this.filmFcW = getI8('film_fc_w');
    this.filmFcB = getF32('film_fc_b');
    this.finalNormW = getF32('final_norm_w');

    this.layers = [];
    for (let l = 0; l < this.cfg.n_layers; l++) {
      const p = `layers.${l}.`;
      this.layers.push({
        norm1_w: getF32(`${p}norm1_w`),
        norm2_w: getF32(`${p}norm2_w`),
        film_w: getI8(`${p}film_w`),
        film_b: getF32(`${p}film_b`),
        q_w: getI8(`${p}q_w`),
        k_w: getI8(`${p}k_w`),
        v_w: getI8(`${p}v_w`),
        o_w: getI8(`${p}o_w`),
        w1_w: getI8(`${p}w1_w`),
        w3_w: getI8(`${p}w3_w`),
        w2_w: getI8(`${p}w2_w`),
      });
    }

    // Precompute RoPE cos/sin tables for [max_seq_len, head_dim / 2]
    const half = this.cfg.head_dim / 2;
    this.ropeCos = new Float32Array(this.cfg.max_seq_len * half);
    this.ropeSin = new Float32Array(this.cfg.max_seq_len * half);
    for (let pos = 0; pos < this.cfg.max_seq_len; pos++) {
      for (let i = 0; i < half; i++) {
        const invFreq = 1 / this.cfg.rope_base ** ((2 * i) / this.cfg.head_dim);
        const angle = pos * invFreq;
        this.ropeCos[pos * half + i] = Math.cos(angle);
        this.ropeSin[pos * half + i] = Math.sin(angle);
      }
    }
  }

  private computeConditioning(z: Float32Array): {
    prefix: Float32Array;
    films: Float32Array[];
  } {
    const { d_model, cond_dim, film_dim, prefix_len, n_layers, rms_eps } = this.cfg;
    let normSq = 0;
    for (let i = 0; i < z.length; i++) normSq += z[i] * z[i];

    if (Math.sqrt(normSq) < 1e-6) {
      const films: Float32Array[] = [];
      for (let l = 0; l < n_layers; l++) films.push(new Float32Array(4 * d_model));
      return { prefix: this.nullPrefix, films };
    }

    const h1 = new Float32Array(cond_dim);
    matvecInt8(h1, this.pocketFc1W, z, this.pocketFc1B);
    for (let i = 0; i < cond_dim; i++) h1[i] = silu(h1[i]);

    const h2 = new Float32Array(cond_dim);
    matvecInt8(h2, this.pocketFc2W, h1, this.pocketFc2B);
    for (let i = 0; i < cond_dim; i++) h2[i] = silu(h2[i]);

    const rawPref = new Float32Array(prefix_len * d_model);
    matvecInt8(rawPref, this.prefixProjW, h2, this.prefixProjB);
    const prefix = new Float32Array(prefix_len * d_model);
    for (let p = 0; p < prefix_len; p++) {
      const sliceIn = rawPref.subarray(p * d_model, (p + 1) * d_model);
      const sliceOut = prefix.subarray(p * d_model, (p + 1) * d_model);
      rmsNorm(sliceOut, sliceIn, this.prefixNormW, rms_eps);
    }

    const filmH = new Float32Array(film_dim);
    matvecInt8(filmH, this.filmFcW, h2, this.filmFcB);
    for (let i = 0; i < film_dim; i++) filmH[i] = silu(filmH[i]);

    const films: Float32Array[] = [];
    for (let l = 0; l < n_layers; l++) {
      const fl = new Float32Array(4 * d_model);
      matvecInt8(fl, this.layers[l].film_w, filmH, this.layers[l].film_b);
      films.push(fl);
    }
    return { prefix, films };
  }

  private applyRopeInPlace(vec: Float32Array, pos: number): void {
    const { n_heads, head_dim } = this.cfg;
    const half = head_dim / 2;
    const baseOff = pos * half;
    for (let h = 0; h < n_heads; h++) {
      const hOff = h * head_dim;
      for (let i = 0; i < half; i++) {
        const c = this.ropeCos[baseOff + i];
        const s = this.ropeSin[baseOff + i];
        const x1 = vec[hOff + i];
        const x2 = vec[hOff + half + i];
        vec[hOff + i] = x1 * c - x2 * s;
        vec[hOff + half + i] = x2 * c + x1 * s;
      }
    }
  }

  private createStream(z: Float32Array) {
    const { d_model, n_layers, n_heads, head_dim, d_ff, max_seq_len, rms_eps, vocab_size } =
      this.cfg;
    const { prefix, films } = this.computeConditioning(z);
    const kCache = Array.from({ length: n_layers }, () => new Float32Array(max_seq_len * d_model));
    const vCache = Array.from({ length: n_layers }, () => new Float32Array(max_seq_len * d_model));
    const x = new Float32Array(d_model);
    const normBuf = new Float32Array(d_model);
    const q = new Float32Array(d_model);
    const k = new Float32Array(d_model);
    const v = new Float32Array(d_model);
    const attnOut = new Float32Array(d_model);
    const oOut = new Float32Array(d_model);
    const scores = new Float32Array(max_seq_len);
    const g1 = new Float32Array(d_ff);
    const g3 = new Float32Array(d_ff);
    const ffnOut = new Float32Array(d_model);
    const logits = new Float32Array(vocab_size);
    const invSqrtD = 1 / Math.sqrt(head_dim);

    const forwardVec = (inputVec: Float32Array, pos: number, computeLogits: boolean) => {
      x.set(inputVec);
      for (let l = 0; l < n_layers; l++) {
        const lw = this.layers[l];
        const fl = films[l];

        // 1. Pre-norm + FiLM
        rmsNorm(normBuf, x, lw.norm1_w, rms_eps);
        for (let i = 0; i < d_model; i++) {
          normBuf[i] = normBuf[i] * (1 + fl[i]) + fl[d_model + i];
        }

        // 2. QKV + RoPE + KV-cache causal attention
        matvecInt8(q, lw.q_w, normBuf);
        matvecInt8(k, lw.k_w, normBuf);
        matvecInt8(v, lw.v_w, normBuf);
        this.applyRopeInPlace(q, pos);
        this.applyRopeInPlace(k, pos);

        const cacheOff = pos * d_model;
        kCache[l].set(k, cacheOff);
        vCache[l].set(v, cacheOff);

        const seqLen = pos + 1;
        for (let h = 0; h < n_heads; h++) {
          const hOff = h * head_dim;
          let maxS = -Infinity;
          for (let t = 0; t < seqLen; t++) {
            const ktOff = t * d_model + hOff;
            let dot = 0;
            for (let d = 0; d < head_dim; d++) dot += q[hOff + d] * kCache[l][ktOff + d];
            const sc = dot * invSqrtD;
            scores[t] = sc;
            if (sc > maxS) maxS = sc;
          }
          let sumExp = 0;
          for (let t = 0; t < seqLen; t++) {
            const e = Math.exp(scores[t] - maxS);
            scores[t] = e;
            sumExp += e;
          }
          const invSum = 1 / sumExp;
          for (let d = 0; d < head_dim; d++) {
            let acc = 0;
            for (let t = 0; t < seqLen; t++) {
              acc += scores[t] * invSum * vCache[l][t * d_model + hOff + d];
            }
            attnOut[hOff + d] = acc;
          }
        }

        matvecInt8(oOut, lw.o_w, attnOut);
        for (let i = 0; i < d_model; i++) x[i] += oOut[i];

        // 3. FFN pre-norm + FiLM + SwiGLU
        rmsNorm(normBuf, x, lw.norm2_w, rms_eps);
        for (let i = 0; i < d_model; i++) {
          normBuf[i] = normBuf[i] * (1 + fl[2 * d_model + i]) + fl[3 * d_model + i];
        }
        matvecInt8(g1, lw.w1_w, normBuf);
        matvecInt8(g3, lw.w3_w, normBuf);
        for (let i = 0; i < d_ff; i++) g1[i] = silu(g1[i]) * g3[i];
        matvecInt8(ffnOut, lw.w2_w, g1);
        for (let i = 0; i < d_model; i++) x[i] += ffnOut[i];
      }

      if (computeLogits) {
        rmsNorm(normBuf, x, this.finalNormW, rms_eps);
        for (let vIdx = 0; vIdx < vocab_size; vIdx++) {
          let dot = 0;
          const eOff = vIdx * d_model;
          for (let i = 0; i < d_model; i++) dot += this.tokEmb[eOff + i] * normBuf[i];
          logits[vIdx] = dot + this.lmHeadBias[vIdx];
        }
      }
      return logits;
    };

    return {
      prefill: () => {
        for (let p = 0; p < this.cfg.prefix_len; p++) {
          forwardVec(prefix.subarray(p * d_model, (p + 1) * d_model), p, false);
        }
        const bosVec = this.tokEmb.subarray(this.cfg.bos_id * d_model, (this.cfg.bos_id + 1) * d_model);
        return forwardVec(bosVec, this.cfg.prefix_len, true);
      },
      step: (tokenId: number, pos: number) => {
        const vec = this.tokEmb.subarray(tokenId * d_model, (tokenId + 1) * d_model);
        return forwardVec(vec, pos, true);
      },
    };
  }

  generateOne(opts: GenerateOptions): GeneratedSample {
    const t0 = performance.now();
    const rng = opts.seed !== undefined ? mulberry32(opts.seed) : Math.random;
    const { vocab_size, pad_id, bos_id, eos_id, unk_id, prefix_len, max_seq_len } = this.cfg;
    const useCfg = Math.abs(opts.cfgScale - 1.0) > 1e-4;

    const condStream = this.createStream(opts.zCond);
    const uncondStream = useCfg ? this.createStream(new Float32Array(this.cfg.pocket_dim)) : null;

    let condLogits = condStream.prefill();
    let uncondLogits = uncondStream ? uncondStream.prefill() : null;

    const guided = new Float32Array(vocab_size);
    const indices = Array.from({ length: vocab_size }, (_, i) => i);
    const probs = new Float32Array(vocab_size);
    const tokens: string[] = [];

    const maxSteps = Math.min(opts.maxNewTokens, max_seq_len - prefix_len - 1);
    const invTemp = 1 / Math.max(opts.temperature, 1e-4);

    for (let step = 0; step < maxSteps; step++) {
      for (let i = 0; i < vocab_size; i++) {
        if (i === pad_id || i === bos_id || i === unk_id) {
          guided[i] = -Infinity;
        } else if (useCfg && uncondLogits) {
          guided[i] = (uncondLogits[i] + opts.cfgScale * (condLogits[i] - uncondLogits[i])) * invTemp;
        } else {
          guided[i] = condLogits[i] * invTemp;
        }
      }

      indices.sort((a, b) => guided[b] - guided[a]);
      const kLimit = Math.min(opts.topK > 0 ? opts.topK : vocab_size, vocab_size);
      const maxLogit = guided[indices[0]];

      let sumExp = 0;
      for (let i = 0; i < kLimit; i++) {
        const val = guided[indices[i]];
        const e = Number.isFinite(val) ? Math.exp(val - maxLogit) : 0;
        probs[i] = e;
        sumExp += e;
      }

      // Top-p nucleus truncation
      let cutoff = kLimit;
      let cum = 0;
      for (let i = 0; i < kLimit; i++) {
        probs[i] /= sumExp;
        cum += probs[i];
        if (cum >= opts.topP && i >= 0) {
          cutoff = i + 1;
          break;
        }
      }

      let r = rng() * cum;
      let chosen = indices[0];
      for (let i = 0; i < cutoff; i++) {
        r -= probs[i];
        if (r <= 0) {
          chosen = indices[i];
          break;
        }
      }

      if (chosen === eos_id) break;
      const tok = this.manifest.vocab[chosen] ?? '';
      tokens.push(tok);
      if (opts.onToken) opts.onToken(tokens.join(''));

      const nextPos = prefix_len + 1 + step;
      if (nextPos >= max_seq_len) break;
      condLogits = condStream.step(chosen, nextPos);
      if (uncondStream) uncondLogits = uncondStream.step(chosen, nextPos);
    }

    return {
      smiles: tokens.join(''),
      tokenCount: tokens.length,
      latencyMs: performance.now() - t0,
    };
  }

  async generateOneAsync(opts: GenerateOptions): Promise<GeneratedSample> {
    const t0 = performance.now();
    const rng = opts.seed !== undefined ? mulberry32(opts.seed) : Math.random;
    const { vocab_size, pad_id, bos_id, eos_id, unk_id, prefix_len, max_seq_len } = this.cfg;
    const useCfg = Math.abs(opts.cfgScale - 1.0) > 1e-4;

    const condStream = this.createStream(opts.zCond);
    const uncondStream = useCfg ? this.createStream(new Float32Array(this.cfg.pocket_dim)) : null;

    let condLogits = condStream.prefill();
    let uncondLogits = uncondStream ? uncondStream.prefill() : null;

    const guided = new Float32Array(vocab_size);
    const indices = Array.from({ length: vocab_size }, (_, i) => i);
    const probs = new Float32Array(vocab_size);
    const tokens: string[] = [];

    const maxSteps = Math.min(opts.maxNewTokens, max_seq_len - prefix_len - 1);
    const invTemp = 1 / Math.max(opts.temperature, 1e-4);

    for (let step = 0; step < maxSteps; step++) {
      for (let i = 0; i < vocab_size; i++) {
        if (i === pad_id || i === bos_id || i === unk_id) {
          guided[i] = -Infinity;
        } else if (useCfg && uncondLogits) {
          guided[i] =
            (uncondLogits[i] + opts.cfgScale * (condLogits[i] - uncondLogits[i])) * invTemp;
        } else {
          guided[i] = condLogits[i] * invTemp;
        }
      }

      indices.sort((a, b) => guided[b] - guided[a]);
      const kLimit = Math.min(opts.topK > 0 ? opts.topK : vocab_size, vocab_size);
      const maxLogit = guided[indices[0]];

      let sumExp = 0;
      for (let i = 0; i < kLimit; i++) {
        const val = guided[indices[i]];
        const e = Number.isFinite(val) ? Math.exp(val - maxLogit) : 0;
        probs[i] = e;
        sumExp += e;
      }

      let cutoff = kLimit;
      let cum = 0;
      for (let i = 0; i < kLimit; i++) {
        probs[i] /= sumExp;
        cum += probs[i];
        if (cum >= opts.topP && i >= 0) {
          cutoff = i + 1;
          break;
        }
      }

      let r = rng() * cum;
      let chosen = indices[0];
      for (let i = 0; i < cutoff; i++) {
        r -= probs[i];
        if (r <= 0) {
          chosen = indices[i];
          break;
        }
      }

      if (chosen === eos_id) break;
      const tok = this.manifest.vocab[chosen] ?? '';
      tokens.push(tok);
      if (opts.onToken && (step % 4 === 0 || step === maxSteps - 1)) {
        opts.onToken(tokens.join(''));
      }
      if (step > 0 && step % 12 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      const nextPos = prefix_len + 1 + step;
      if (nextPos >= max_seq_len) break;
      condLogits = condStream.step(chosen, nextPos);
      if (uncondStream) uncondLogits = uncondStream.step(chosen, nextPos);
    }

    return {
      smiles: tokens.join(''),
      tokenCount: tokens.length,
      latencyMs: performance.now() - t0,
    };
  }
}

let cachedBundle: Promise<{
  transformer: BrowserChemicalTransformer;
  pockets: PocketEmbeddingsPack;
}> | null = null;

export function loadBrowserGeneratorBundle(): Promise<{
  transformer: BrowserChemicalTransformer;
  pockets: PocketEmbeddingsPack;
}> {
  if (cachedBundle) return cachedBundle;
  cachedBundle = (async () => {
    const [manifestRes, pocketsRes] = await Promise.all([
      fetch(assetUrl('models/generator_manifest.json')),
      fetch(assetUrl('models/pocket_embeddings.json')),
    ]);
    if (!manifestRes.ok || !pocketsRes.ok) {
      throw new Error('Browser neural model pack not found in web/public/models/.');
    }
    const manifest = (await manifestRes.json()) as GeneratorManifest;
    const pockets = (await pocketsRes.json()) as PocketEmbeddingsPack;
    const weightsRes = await fetch(assetUrl(`models/${manifest.weights_file}`));
    if (!weightsRes.ok) {
      throw new Error(`Missing generator weights binary: ${manifest.weights_file}`);
    }
    const buf = await weightsRes.arrayBuffer();
    const transformer = new BrowserChemicalTransformer(manifest, buf);
    return { transformer, pockets };
  })();
  cachedBundle.catch(() => {
    cachedBundle = null;
  });
  return cachedBundle;
}
