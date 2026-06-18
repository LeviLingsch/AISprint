// depth-worker.js — RELATIVE depth streamed directly from HuggingFace (no bundled model).
// Streams onnx-community/depth-anything-v2-small (dynamic input) and runs it LIVE on the
// phone at a coarse input size. That model's output is RELATIVE inverse-depth (higher =
// nearer); we per-frame normalize it and map to a pseudo-distance the AISprint 7-sector
// synth consumes exactly like meters.
//
// URL flags: ?size=NNN (model input, multiple of 14; smaller=faster, default 252),
//            ?backend=wasm|webgpu (force a backend), ?fresh=1 (bypass model cache).

import { pipeline, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

env.allowLocalModels = false;                 // always stream the weights from the HF hub
if (new URLSearchParams(self.location.search).has('fresh')) env.useBrowserCache = false;
try { env.backends.onnx.wasm.numThreads = Math.min(8, (self.navigator && self.navigator.hardwareConcurrency) || 4); } catch {}

const MODEL = 'onnx-community/depth-anything-v2-small';
const SP = new URLSearchParams(self.location.search);
const FORCE = SP.get('backend') || undefined;
let SIZE = parseInt(SP.get('size'), 10);
if (!Number.isFinite(SIZE) || SIZE % 14 !== 0) SIZE = 252;    // default coarse/fast size (18*14)

let estimator = null, backend = '';

async function load() {
  if (estimator) return backend;
  const prog = (p) => { if (p.status === 'progress' && p.file) self.postMessage({ type: 'progress', file: p.file, pct: Math.round(p.progress || 0) }); };
  if (FORCE !== 'wasm') {
    try { estimator = await pipeline('depth-estimation', MODEL, { device: 'webgpu', dtype: 'fp16', progress_callback: prog }); backend = 'webgpu'; setSize(estimator, SIZE); return backend; }
    catch (e) { self.postMessage({ type: 'info', message: 'WebGPU unavailable -> WASM' }); }
  }
  estimator = await pipeline('depth-estimation', MODEL, { device: 'wasm', dtype: 'q8', progress_callback: prog });
  backend = 'wasm'; setSize(estimator, SIZE); return backend;
}

// best-effort: shrink the model input on this dynamic model (the speed lever)
function setSize(est, size) {
  for (const ip of [est && est.processor, est && est.processor && est.processor.image_processor, est && est.processor && est.processor.feature_extractor]) {
    if (ip && ip.size) { try { ip.size = { width: size, height: size }; ip.do_resize = true; if ('keep_aspect_ratio' in ip) ip.keep_aspect_ratio = false; if ('ensure_multiple_of' in ip) ip.ensure_multiple_of = 14; } catch {} }
  }
}

function percentile(sorted, p) { const idx = (p / 100) * (sorted.length - 1), lo = Math.floor(idx), hi = Math.ceil(idx); return lo === hi ? sorted[lo] : sorted[lo] * (hi - idx) + sorted[hi] * (idx - lo); }

// relative DA output: HIGHER = NEARER. Per-frame normalize -> closeness -> pseudo-distance.
function sectorDistances(data, H, W, near_m, far_m, n = 7, rt = 0.10, rb = 0.80) {
  const top = Math.floor(rt * H), bot = Math.floor(rb * H);
  let mn = Infinity, mx = -Infinity;
  for (let y = top; y < bot; y++) { const row = y * W; for (let x = 0; x < W; x++) { const v = data[row + x]; if (v < mn) mn = v; if (v > mx) mx = v; } }
  const inv = 1 / (mx - mn + 1e-6), out = new Float32Array(n);
  for (let s = 0; s < n; s++) {
    const x0 = Math.floor(s * W / n), x1 = Math.floor((s + 1) * W / n), vals = [];
    for (let y = top; y < bot; y++) { const row = y * W; for (let x = x0; x < x1; x++) vals.push(data[row + x]); }
    vals.sort((a, b) => a - b);
    const nearVal = percentile(vals, 80);                 // higher = nearer -> nearest obstacle
    const c = Math.max(0, Math.min(1, (nearVal - mn) * inv)); // closeness 0..1 (1 = nearest)
    out[s] = far_m - c * (far_m - near_m);                 // pseudo-distance, smaller = nearer (synth-compatible)
  }
  return out;
}

function turbo(t) { const r = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 3), 0), 1)); const g = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 2), 0), 1)); const b = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 1), 0), 1)); return [r, g, b]; }
function depthThumb(data, H, W, DW = 96, DH = 72) {
  let mn = Infinity, mx = -Infinity; for (let i = 0; i < data.length; i += 7) { const v = data[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const inv = 1 / (mx - mn + 1e-6), rgba = new Uint8ClampedArray(DW * DH * 4);
  for (let y = 0; y < DH; y++) { const sy = Math.floor(y * H / DH); for (let x = 0; x < DW; x++) { const sx = Math.floor(x * W / DW);
    const t = (data[sy * W + sx] - mn) * inv;             // relative: higher = nearer = hot
    const [r, g, b] = turbo(t); const o = (y * DW + x) * 4; rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = 255; } }
  return { rgba, DW, DH };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'infer') {
      const be = await load();
      const img = new RawImage(new Uint8ClampedArray(m.buf), m.width, m.height, 4).rgb();
      const out = await estimator(img);
      const pd = out.predicted_depth, dims = pd.dims;
      const H = dims.length === 3 ? dims[1] : dims[0];
      const W = dims.length === 3 ? dims[2] : dims[1];
      const near = (typeof m.near === 'number') ? m.near : 0.5;   // guard: don't NaN if nav.js is stale
      const far = (typeof m.far === 'number') ? m.far : 3.0;
      const dist = sectorDistances(pd.data, H, W, near, far);
      const th = depthThumb(pd.data, H, W);
      self.postMessage({ type: 'dist', dist: Array.from(dist), depth: th.rgba.buffer, dw: th.DW, dh: th.DH, backend: be, res: `${W}x${H}` }, [th.rgba.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
