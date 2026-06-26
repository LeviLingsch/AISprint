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
const FORCE = SP.get('backend') || undefined; // default: try WebGPU first (override with ?backend=wasm)
const sizeArg = parseInt(SP.get('size'), 10);
const SIZE = (Number.isFinite(sizeArg) && sizeArg % 14 === 0) ? sizeArg : null; // null = model's native size (reliable). ?size=126 to experiment.

let estimator = null, backend = '';

async function load() {
  if (estimator) return backend;
  const prog = (p) => { if (p.status === 'progress' && p.file) self.postMessage({ type: 'progress', file: p.file, pct: Math.round(p.progress || 0) }); };
  if (FORCE !== 'wasm') {
    try { estimator = await pipeline('depth-estimation', MODEL, { device: 'webgpu', dtype: 'fp16', progress_callback: prog }); backend = 'webgpu'; if (SIZE) setSize(estimator, SIZE); return backend; }
    catch (e) { self.postMessage({ type: 'info', message: 'WebGPU init failed: ' + ((e && e.message) || e) }); if (FORCE === 'webgpu') throw e; } // forced webgpu: NO wasm fallback
  }
  estimator = await pipeline('depth-estimation', MODEL, { device: 'wasm', dtype: 'q8', progress_callback: prog });
  backend = 'wasm'; if (SIZE) setSize(estimator, SIZE); return backend;
}

// best-effort: shrink the model input on this dynamic model (the speed lever)
function setSize(est, size) {
  for (const ip of [est && est.processor, est && est.processor && est.processor.image_processor, est && est.processor && est.processor.feature_extractor]) {
    if (ip && ip.size) { try { ip.size = { width: size, height: size }; ip.do_resize = true; if ('keep_aspect_ratio' in ip) ip.keep_aspect_ratio = false; if ('ensure_multiple_of' in ip) ip.ensure_multiple_of = 14; } catch {} }
  }
}

function percentile(sorted, p) { const idx = (p / 100) * (sorted.length - 1), lo = Math.floor(idx), hi = Math.ceil(idx); return lo === hi ? sorted[lo] : sorted[lo] * (hi - idx) + sorted[hi] * (idx - lo); }

// relative DA output: HIGHER = NEARER (inverse-depth), no real meters and a scale that
// varies by scene. Convert to a pseudo-distance d = SCALE / value so genuinely-far regions
// map BEYOND far_m and the synth SILENCES them (restores "quiet unless something is close",
// NOT per-frame normalized so an open scene stays silent). SCALE is anchored to NEAR_REF
// (the raw value of a ~near_m object): auto-calibrates from the closest thing seen in the
// first ~12 frames; override with ?nearref=VALUE (read the live "ref" off the status line).
// ANCHOR_M = assumed real distance (m) of the nearest thing in view at startup. Pseudo-distance
// is ANCHOR_M * ref / value, so a normal room view (nearest ~1.5 m) puts the silence boundary
// near far_m (3 m). RAISE ?anchor to tighten the audible range (more silence), lower to widen.
const ANCHOR_M = parseFloat(SP.get('anchor')) || 1.6;
let NEAR_REF = parseFloat(SP.get('nearref')) || 0;
const AUTO_REF = !(NEAR_REF > 0);
let calibSum = 0, calibN = 0;

function sectorDistances(data, H, W, near_m, far_m, n = 7, rt = 0.10, rb = 0.80) {
  const top = Math.floor(rt * H), bot = Math.floor(rb * H);
  const sv = new Float32Array(n);
  let frameNear = 0;
  for (let s = 0; s < n; s++) {
    const x0 = Math.floor(s * W / n), x1 = Math.floor((s + 1) * W / n), vals = [];
    for (let y = top; y < bot; y++) { const row = y * W; for (let x = x0; x < x1; x++) vals.push(data[row + x]); }
    vals.sort((a, b) => a - b);
    sv[s] = percentile(vals, 80);                          // nearest obstacle in this sector (higher=nearer)
    if (sv[s] > frameNear) frameNear = sv[s];
  }
  if (AUTO_REF && calibN < 12) { calibSum += frameNear; calibN++; NEAR_REF = calibSum / calibN; } // avg first frames (stable)
  const ref = NEAR_REF || frameNear || 1;
  const scale = ANCHOR_M * ref;                            // nearest startup thing -> ~ANCHOR_M meters
  const out = new Float32Array(n);
  for (let s = 0; s < n; s++) out[s] = scale / Math.max(sv[s], 1e-4); // far -> large -> > far_m -> SILENT
  return { out, ref };
}

function turbo(t) { const r = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 3), 0), 1)); const g = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 2), 0), 1)); const b = Math.round(255 * Math.min(Math.max(1.5 - Math.abs(4 * t - 1), 0), 1)); return [r, g, b]; }
function depthThumb(data, H, W, camW, camH) {
  // thumbnail at the CAMERA's aspect ratio so the preview matches what the camera sees
  // (not stretched flat). Sample the full depth map onto that grid.
  const aw = camW || W, ah = camH || H, LONG = 144;
  const DW = aw >= ah ? LONG : Math.max(1, Math.round(LONG * aw / ah));
  const DH = aw >= ah ? Math.max(1, Math.round(LONG * ah / aw)) : LONG;
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
      const sd = sectorDistances(pd.data, H, W, near, far);
      const th = depthThumb(pd.data, H, W, m.width, m.height);
      self.postMessage({ type: 'dist', dist: Array.from(sd.out), ref: sd.ref, depth: th.rgba.buffer, dw: th.DW, dh: th.DH, backend: be, res: `${W}x${H}` }, [th.rgba.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
