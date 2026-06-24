// synth-worklet.js — faithful Web Audio port of AISprint/src/synth.py (Levi's engine).
//
// Cue mapping (identical to the Python desktop demo):
//   loudness  = proximity      (close obstacle = loud, open = silent)
//   pan       = left / right   (constant-power)
//   frequency = centrality     (center = center_hz HIGH, edges = side_hz LOW)
// Three modes: continuous / pulse / sweep. Phase-continuous so params can change
// every frame without clicks. Runs on the audio thread; depth feeds it via messages.

// Each voice is a rich, slightly-detuned organ/pad tone (NOT a single sine):
// [freq-ratio, amplitude] — a fifth + octaves give chordal body, and the ~0.5%
// detunes add a warm chorus/shimmer. Makes every sector "bar" a fuller instrument note.
// Warm harmonic voice — NO inharmonic fifth (that 1.5x partial was the buzz). Harmonics
// above the fundamental are scaled per-voice by "brightness": LOW notes (the sides) stay
// nearly pure and smooth, higher notes (toward center) pick up a little more life.
const PARTIALS = [
  [1.0, 1.00],   // fundamental
  [2.0, 0.40],   // octave (warmth/body)
  [3.0, 0.14],   // octave + fifth
  [4.0, 0.06],   // two octaves (faint sparkle, mostly on the high notes)
];
const PN = PARTIALS.reduce((s, p) => s + p[1], 0);
const brightOf = (f) => Math.min(1, Math.max(0.15, (f - 200) / 600));  // low freq -> fewer harmonics
// Per-sector pitch snaps to a consonant chord (semitones above the root, edge -> center):
// root, fifth, octave, octave+major-third -> overlapping sectors form a major chord.
// Other pleasant options: [0,7,12,19] open fifths · [0,3,7,12] minor · [0,5,7,12] sus4.
const CHORD = [0, 7, 12, 16];

function timbre(phase, bright) {
  let out = Math.sin(phase);                                    // fundamental, full
  for (let i = 1; i < PARTIALS.length; i++) out += bright * PARTIALS[i][1] * Math.sin(PARTIALS[i][0] * phase);
  return out / PN;
}
const clip01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

// linear interpolate gain at azimuth caz over ascending az[] (mirrors np.interp)
function interp(caz, az, gain) {
  const n = az.length;
  if (caz <= az[0]) return gain[0];
  if (caz >= az[n - 1]) return gain[n - 1];
  let i = 1;
  while (i < n && az[i] < caz) i++;
  const t = (caz - az[i - 1]) / (az[i] - az[i - 1]);
  return gain[i - 1] + t * (gain[i] - gain[i - 1]);
}

class EchoSynth extends AudioWorkletProcessor {
  constructor() {
    super();
    const N = 7;
    this.N = N;
    this.near_m = 0.5; this.far_m = 3.0;
    this.side_hz = 300.0; this.center_hz = 800.0;
    this.falloff = 2.0;
    this.rate_min = 1.6; this.rate_max = 11.0;
    this.sweep_period = 0.3;
    this.smooth_hz = 3.0;   // gentler -> volume doesn't jitter with noisy live depth
    this.master = 0.8;
    this.mode = 'continuous';

    this._geom();
    // streaming state
    this.phase = new Float64Array(N);
    this.pulse_phase = new Float64Array(N);
    this.sweep_phase = 0.0;
    this.sweep_cphase = 0.0;
    this.smooth_d = new Float64Array(N).fill(this.far_m);
    this.last_gain = new Float64Array(N);
    this.target_d = new Float64Array(N).fill(this.far_m);

    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.dist) { const d = m.dist; for (let k = 0; k < N; k++) this.target_d[k] = d[k]; return; }
      if (m.params) {
        const p = m.params;
        for (const key of ['near_m', 'far_m', 'side_hz', 'center_hz', 'falloff', 'sweep_period', 'master']) {
          if (p[key] != null) this[key] = p[key];
        }
        if (p.mode) this.mode = p.mode;
        this._geom();
      }
    };
  }

  _geom() {
    const N = this.N;
    this.az = new Float64Array(N);
    this.gL = new Float64Array(N);
    this.gR = new Float64Array(N);
    this.freq = new Float64Array(N);
    this.bright = new Float64Array(N);
    // span the FULL stereo width: leftmost column -> -1 (hard left), rightmost -> +1
    for (let i = 0; i < N; i++) this.az[i] = N > 1 ? (i / (N - 1)) * 2 - 1 : 0;
    // distinct |az| levels (center smallest .. edge largest); map each to a chord tone
    const absA = Array.from(this.az, (a) => Math.abs(a));
    const levels = Array.from(new Set(absA.map((v) => v.toFixed(4)))).map(Number).sort((a, b) => a - b);
    const L = levels.length;
    const root = this.side_hz;                          // sides/edges = chord root (low)
    for (let i = 0; i < N; i++) {
      const theta = (this.az[i] + 1) / 2 * (Math.PI / 2);
      this.gL[i] = Math.cos(theta); this.gR[i] = Math.sin(theta);
      const lvl = levels.indexOf(Number(absA[i].toFixed(4)));         // 0 = center, L-1 = edge
      const ci = L > 1 ? Math.round((L - 1 - lvl) / (L - 1) * (CHORD.length - 1)) : CHORD.length - 1;
      this.freq[i] = root * Math.pow(2, CHORD[ci] / 12);              // edge=root .. center=top of chord
      this.bright[i] = brightOf(this.freq[i]);
    }
    this.center_hz = root * Math.pow(2, CHORD[CHORD.length - 1] / 12); // keep the sweep glide consistent
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const Lout = out[0], Rout = out[1] || out[0];
    const n = Lout.length;                                // 128
    const sr = sampleRate, N = this.N;

    const alpha = 1 - Math.exp(-2 * Math.PI * this.smooth_hz * n / sr);
    const prox = new Float64Array(N), gain = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      this.smooth_d[k] += alpha * (this.target_d[k] - this.smooth_d[k]);
      const p = clip01((this.far_m - this.smooth_d[k]) / (this.far_m - this.near_m));
      prox[k] = p; gain[k] = Math.pow(p, this.falloff);
    }

    for (let i = 0; i < n; i++) { Lout[i] = 0; Rout[i] = 0; }

    if (this.mode === 'continuous' || this.mode === 'pulse') {
      const pulse = this.mode === 'pulse';
      for (let k = 0; k < N; k++) {
        const inc = 2 * Math.PI * this.freq[k] / sr;
        const g0 = this.last_gain[k], g1 = gain[k];
        const rate = this.rate_min + prox[k] * (this.rate_max - this.rate_min);
        const pinc = rate / sr;
        const duty = Math.min(0.9, Math.max(0.05, 0.085 * rate));
        const gL = this.gL[k], gR = this.gR[k], brightK = this.bright[k];
        let ph = this.phase[k], pph = this.pulse_phase[k];
        for (let i = 0; i < n; i++) {
          const j = i + 1;
          const g = g0 + (g1 - g0) * (j / n);             // linspace last->current
          let sig = timbre(ph + inc * j, brightK) * g;
          if (pulse) {
            const frac = (pph + pinc * j) % 1.0;
            const env = frac < duty ? 0.5 - 0.5 * Math.cos(2 * Math.PI * frac / duty) : 0.0;
            sig *= env;
          }
          Lout[i] += sig * gL;
          Rout[i] += sig * gR;
        }
        this.phase[k] = (ph + inc * n) % (2 * Math.PI);
        this.pulse_phase[k] = (pph + pinc * n) % 1.0;
      }
      for (let k = 0; k < N; k++) this.last_gain[k] = gain[k];

    } else if (this.mode === 'sweep') {
      const sinc = 1.0 / (this.sweep_period * sr);
      const side = this.side_hz, center = this.center_hz;
      let sph = this.sweep_phase, cph = this.sweep_cphase;
      for (let i = 0; i < n; i++) {
        const j = i + 1;
        const cursor01 = (sph + sinc * j) % 1.0;
        const caz = cursor01 * 2 - 1;
        let g_at = interp(caz, this.az, gain);
        const edge = clip01(Math.min(cursor01, 1 - cursor01) / 0.04); // no wrap click
        g_at *= edge;
        const freq_at = center * Math.pow(side / center, Math.abs(caz));
        cph += 2 * Math.PI * freq_at / sr;
        const sig = timbre(cph, brightOf(freq_at)) * g_at * 1.7;   // single source vs chord
        const theta = (caz + 1) / 2 * (Math.PI / 2);
        Lout[i] += sig * Math.cos(theta);
        Rout[i] += sig * Math.sin(theta);
      }
      this.sweep_phase = (sph + sinc * n) % 1.0;
      this.sweep_cphase = cph % (2 * Math.PI);
    }

    // Headroom: when many sectors are loud at once their voices stack and slam the
    // limiter — that distortion is the "rough" sound. Scale down by the total gain so
    // a busy scene stays clean (closer is still relatively louder within the scene).
    let norm = this.master;
    if (this.mode === 'continuous' || this.mode === 'pulse') {
      let gsum = 0; for (let k = 0; k < N; k++) gsum += gain[k];
      norm = this.master / (1 + 0.4 * gsum);
    }
    for (let i = 0; i < n; i++) {
      Lout[i] = Math.tanh(Lout[i] * norm);
      Rout[i] = Math.tanh(Rout[i] * norm);
    }
    return true;
  }
}

registerProcessor('echo-synth', EchoSynth);
