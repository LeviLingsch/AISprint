# AISprint Echolocation — Web app

The AISprint desktop engine (`../src/`) ported to run in a phone browser:
the **same metric depth model** (Depth-Anything-V2-Metric-Indoor-Small) and the
**same sonification** (`synth.py` → `synth-worklet.js`: 7 sectors, loudness=proximity,
pan=L/R, pitch=centrality; continuous / pulse / sweep). Camera + depth + audio run
on-device; no install.

Optional: **double-tap** the screen to ask Claude to describe the scene (needs an
Anthropic key, pasted in-app or behind a proxy). The navigation works fully without it.

## Run / deploy
Serve `web/` over HTTPS (camera needs it). On GitHub Pages: Settings → Pages →
branch `echolocation` → `/ (root)`, then open `…github.io/AISprint/web/`.

Files: `index.html` · `nav.js` (camera→depth→sectors→audio + double-tap) ·
`depth-worker.js` (metric depth + sector_distances + depth thumbnail) ·
`synth-worklet.js` (port of `src/synth.py`) · `claude-discuss.js` (optional Claude layer) ·
`models/` (self-hosted ONNX: fp16 for WebGPU, int8 for WASM).
