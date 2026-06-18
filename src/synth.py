"""Shared streaming sonifier: depth -> stereo audio, used by both the offline
renderer (sonify.py) and the live camera app (live.py).

Cue mapping:
  * loudness  = proximity, in EVERY direction (close obstacle = loud, clear = quiet).
  * panning   = left / right direction.
  * frequency = how CENTRAL a direction is. The two outer edges share a LOW pitch
                (side_hz); the center uses a HIGH pitch (center_hz); symmetric
                left<->right. So a tone "dead ahead" has a distinct high pitch
                that does not blur into the low side tones. That is how you tell
                "wall ahead" (loud HIGH tone) from "open hallway, walls on both
                sides" (loud LOW tone, quiet/absent high tone) -- with a single
                frequency these two collapse to the same centered image.

Three modes to compare:
  * continuous : one sustained tone per sector; closer = louder. Open = silent.
  * pulse      : like continuous but gated into beeps; closer = louder + faster.
  * sweep      : a single tone scans left->right ~every `sweep_period` s; its
                 loudness tracks the obstacle under the cursor; pan follows the
                 cursor and pitch rises to center_hz in the middle.

Tunable knobs (adjustable live in live.py):
  * side_hz / center_hz : pitch of the outer edges / the center.
  * falloff             : exponent of the distance->loudness curve (lower=gentler).
  * far_m               : hard cutoff -- beyond this a direction is silent (= open).
  * sweep_period        : seconds for one left->right scan.

Design notes:
  * Rich (harmonic) timbre, NOT pure sine: pure tones are very hard to localize.
  * Panning is constant-power amplitude (ILD). HRTF/ITD is a future upgrade.
  * render() is block-based and phase-continuous so parameters can change every
    frame without clicks. Safe to call from a real-time audio callback.
"""
import numpy as np

SR = 44100
HARMONICS = np.array([1.0, 0.5, 0.3, 0.18])  # soft saw-ish -> localizes better
_HN = HARMONICS.sum()
MODES = ("continuous", "pulse", "sweep")


def _timbre(phase):
    out = np.zeros_like(phase)
    for k, a in enumerate(HARMONICS, start=1):
        out += a * np.sin(k * phase)
    return out / _HN


def sector_distances(depth, n_sectors=7, roi_top=0.10, roi_bottom=0.80, pct=20):
    """Per-sector nearest-obstacle distance (m) over a ground/ceiling-excluded band."""
    h, w = depth.shape
    top, bot = int(roi_top * h), int(roi_bottom * h)
    band = depth[top:bot, :]
    edges = np.linspace(0, w, n_sectors + 1).astype(int)
    return np.array([float(np.percentile(band[:, edges[i]:edges[i + 1]], pct))
                     for i in range(n_sectors)])


class Synth:
    def __init__(self, n_sectors=7, sr=SR, near_m=0.5, far_m=6.0,
                 side_hz=300.0, center_hz=720.0, falloff=2.0,
                 rate_min=1.6, rate_max=11.0, sweep_period=0.3,
                 smooth_hz=6.0, master=0.8, mode="continuous"):
        self.sr = sr
        self.N = n_sectors
        self.near_m, self.far_m = near_m, far_m
        self.side_hz, self.center_hz = side_hz, center_hz
        self.falloff = falloff
        self.rate_min, self.rate_max = rate_min, rate_max
        self.sweep_period = sweep_period
        self.smooth_hz = smooth_hz
        self.master = master
        self.mode = mode
        i = np.arange(n_sectors)
        self.az = (i + 0.5) / n_sectors * 2 - 1          # sector azimuth -1..1
        theta = (self.az + 1) / 2 * (np.pi / 2)
        self.gL, self.gR = np.cos(theta), np.sin(theta)  # constant-power pan
        self._recompute_freqs()
        # streaming state
        self.phase = np.zeros(n_sectors)
        self.pulse_phase = np.zeros(n_sectors)
        self.sweep_phase = 0.0
        self.sweep_cphase = 0.0
        self.smooth_d = np.full(n_sectors, far_m, float)
        self.last_gain = np.zeros(n_sectors)
        self.target_d = np.full(n_sectors, far_m, float)

    def _recompute_freqs(self):
        # frequency by how central the sector is: center -> center_hz, edges -> side_hz
        amax = float(np.max(np.abs(self.az))) or 1.0
        t = np.abs(self.az) / amax                       # 0 center .. 1 outer edge
        self.freq = self.center_hz * (self.side_hz / self.center_hz) ** t

    # --- called from the camera/main thread ---
    def set_distances(self, d):
        self.target_d = np.asarray(d, float).copy()

    def set_mode(self, m):
        if m in MODES:
            self.mode = m

    def set_freqs(self, side=None, center=None):
        if side is not None:
            self.side_hz = float(np.clip(side, 60, 4000))
        if center is not None:
            self.center_hz = float(np.clip(center, 60, 4000))
        self._recompute_freqs()

    def _gains(self):
        prox = np.clip((self.far_m - self.smooth_d) / (self.far_m - self.near_m), 0, 1)
        return prox, prox ** self.falloff

    # --- called from the audio thread ---
    def render(self, frames):
        n = frames
        idx = np.arange(1, n + 1)
        alpha = 1 - np.exp(-2 * np.pi * self.smooth_hz * n / self.sr)
        self.smooth_d += alpha * (self.target_d - self.smooth_d)
        prox, gain = self._gains()
        freq = self.freq                  # snapshot (main thread may swap it)
        side, center = self.side_hz, self.center_hz
        L, R = np.zeros(n), np.zeros(n)
        mode = self.mode

        if mode in ("continuous", "pulse"):
            rate = self.rate_min + prox * (self.rate_max - self.rate_min)
            for k in range(self.N):
                inc = 2 * np.pi * freq[k] / self.sr
                wave = _timbre(self.phase[k] + inc * idx)
                g = np.linspace(self.last_gain[k], gain[k], n)
                if mode == "pulse":
                    pinc = rate[k] / self.sr
                    frac = (self.pulse_phase[k] + pinc * idx) % 1.0
                    duty = float(np.clip(0.085 * rate[k], 0.05, 0.9))
                    env = np.where(frac < duty,
                                   0.5 - 0.5 * np.cos(2 * np.pi * frac / duty), 0.0)
                    self.pulse_phase[k] = (self.pulse_phase[k] + pinc * n) % 1.0
                    sig = wave * env * g
                else:
                    sig = wave * g
                L += sig * self.gL[k]
                R += sig * self.gR[k]
                self.phase[k] = (self.phase[k] + inc * n) % (2 * np.pi)
            self.last_gain = gain.copy()

        elif mode == "sweep":
            sinc = 1.0 / (self.sweep_period * self.sr)
            cursor01 = (self.sweep_phase + sinc * idx) % 1.0
            caz = cursor01 * 2 - 1
            g_at = np.interp(caz, self.az, gain)
            edge = np.clip(np.minimum(cursor01, 1 - cursor01) / 0.04, 0, 1)  # no wrap click
            g_at = g_at * edge
            freq_at = center * (side / center) ** np.abs(caz)  # high in the middle
            cph = self.sweep_cphase + np.cumsum(2 * np.pi * freq_at / self.sr)
            wave = _timbre(cph)
            sig = wave * g_at * 1.7                             # single source vs chord
            theta = (caz + 1) / 2 * (np.pi / 2)
            L += sig * np.cos(theta)
            R += sig * np.sin(theta)
            self.sweep_phase = (self.sweep_phase + sinc * n) % 1.0
            self.sweep_cphase = float(cph[-1] % (2 * np.pi))

        out = np.stack([L, R], axis=1) * self.master
        return np.tanh(out).astype(np.float32)  # soft limiter, never harsh-clips
