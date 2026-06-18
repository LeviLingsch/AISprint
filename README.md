# Echolocation — depth-to-sound navigation aid

Turns a camera view into **spatial audio** so a visually impaired user can hear
where obstacles are and where the path ahead is open. Uses **metric (absolute,
meters)** monocular depth, so distances are real, not relative.

This repo is the *echolocation* half of the project (the semantic-description
half is separate).

## Setup

Needs a **Python 3.12** environment with: torch, transformers, opencv, sounddevice, scipy, numpy, pillow, matplotlib.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Levi's Mac the ready-to-use interpreter is the **base conda env**:
`/Users/levilingsch/miniconda3/bin/python` (full stack + model already cached).
Do **not** use the `mica` env — it's missing `transformers` and `sounddevice`.

Depth model: `Depth-Anything-V2-Metric-Indoor-Small` (downloads on first run).
Runs on Apple MPS (~30–50 fps) or CPU (~9–14 fps).

## Run live (webcam, real time)

```bash
python3 src/live.py                 # continuous mode (default)
python3 src/live.py --mode sweep
```

**Use headphones.** macOS: first run needs Camera permission for your terminal
(System Settings → Privacy & Security → Camera).

Keys in the preview window:

| key | action |
|-----|--------|
| `1` `2` `3` | continuous / pulse / sweep |
| `[` `]` | range (`far`): nearer / farther cutoff |
| `-` `=` | loudness falloff: gentler / steeper |
| `a` `s` | side (edge) pitch: down / up |
| `k` `l` | center (ahead) pitch: down / up |
| `,` `.` | sweep: slower / faster |
| `q` / Esc | quit |

The preview shows per-sector distances, a loudness bar, a suggested-path arrow,
and the colorized depth map.

## Offline (test images, no camera)

```bash
python3 src/depth.py      # images -> output/depth/*.npy (+ depth viz). Run once.
python3 src/sonify.py     # depth -> output/audio/<name>_<mode>.wav (+ _map.png)
afplay output/audio/IMG_0279_continuous.wav
```

## How it works

1. **Depth** (`src/depth.py`): RGB → metric depth map in meters.
2. **Sector distances** (`src/synth.py`): the view is split into 7 horizontal
   sectors; per sector we take a robust nearest-obstacle distance over a band
   that excludes the floor-at-your-feet and the ceiling.
3. **Sonification** (`src/synth.py`, `Synth`): three modes to compare —
   - **continuous** — sustained tone per sector; closer = louder; open = silent.
   - **pulse** — beeps; closer = louder + faster (parking-sensor cue).
   - **sweep** — one tone scans left→right (~0.3 s); loudness tracks the obstacle
     under the cursor, while pan and pitch follow it. One sound at a time.

   In all modes: loudness = proximity (close = loud, clear = quiet), pan =
   left/right, and **frequency = how central a direction is** — the outer edges
   share a low pitch (`side_hz`), the center uses a high pitch (`center_hz`),
   symmetric L↔R. So "wall ahead" is a loud HIGH tone, while an open hallway is a
   loud LOW tone (the side walls) with a quiet center. That distinct center pitch
   is what lets you tell the two apart under one consistent loudness rule. A rich
   harmonic timbre (not pure sine) is used because pure tones are hard to localize.

## Files

```
src/depth.py    metric depth estimation (batch, offline)
src/synth.py    sector distances + streaming sonifier (shared)
src/sonify.py   offline: depth maps -> WAVs per mode
src/live.py     real-time webcam app
test_images/    source HEIC photos
work/rgb/       upright PNGs used by the pipeline
output/         depth maps, audio, overlays
```
