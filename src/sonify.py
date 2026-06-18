"""Offline renderer: turn cached depth maps (output/depth/*.npy) into WAVs for
each sonification mode, so we can A/B them on the test images without a camera.

Writes output/audio/<name>_<mode>.wav and a <name>_map.png overlay.
Fast loop: tweak synth.py / params, rerun `python3 src/sonify.py`.
"""
import os
import glob
import numpy as np
from scipy.io import wavfile
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import Synth, sector_distances, MODES, SR

ROOT = "/Users/levilingsch/Code/AISprint"
DEPTH_DIR = os.path.join(ROOT, "output/depth")
RGB_DIR = os.path.join(ROOT, "work/rgb")
OUT_DIR = os.path.join(ROOT, "output/audio")

DURATION = 4.0
NEAR_M, FAR_M = 0.5, 6.0
N_SECTORS = 7


def render_offline(mode, dists, duration):
    syn = Synth(n_sectors=N_SECTORS, near_m=NEAR_M, far_m=FAR_M, mode=mode)
    syn.set_distances(dists)
    syn.smooth_d = np.asarray(dists, float).copy()  # start at steady state
    n, block, buf, done = int(duration * syn.sr), 1024, [], 0
    while done < n:
        b = min(block, n - done)
        buf.append(syn.render(b))
        done += b
    return np.concatenate(buf, axis=0)


def save_map(name, depth, dists):
    rgb = np.asarray(Image.open(os.path.join(RGB_DIR, name + ".png")).convert("RGB"))
    h, w = depth.shape
    top, bot = int(0.10 * h), int(0.80 * h)
    edges = np.linspace(0, w, N_SECTORS + 1).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 7))
    ax.imshow(rgb)
    ax.axhline(top, color="w", ls="--", lw=1)
    ax.axhline(bot, color="w", ls="--", lw=1)
    for e in edges:
        ax.axvline(e, color="w", lw=0.8, alpha=0.6)
    for i, d in enumerate(dists):
        prox = float(np.clip((FAR_M - d) / (FAR_M - NEAR_M), 0, 1))
        cx = (edges[i] + edges[i + 1]) / 2
        ax.text(cx, top - 12, f"{d:.1f}m" if prox > 0 else "open",
                color=(1.0, 1 - prox, 1 - prox), ha="center", va="bottom",
                fontsize=10, weight="bold")
    ax.set_title(f"{name}: nearest obstacle per sector"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name + "_map.png"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(DEPTH_DIR, "*.npy"))):
        name = os.path.splitext(os.path.basename(f))[0]
        depth = np.load(f)
        dists = sector_distances(depth, n_sectors=N_SECTORS)
        save_map(name, depth, dists)
        for mode in MODES:
            stereo = render_offline(mode, dists, DURATION)
            wavfile.write(os.path.join(OUT_DIR, f"{name}_{mode}.wav"),
                          SR, (stereo * 32767).astype(np.int16))
        cells = " ".join((f"{d:4.1f}m" if d < FAR_M else " open") for d in dists)
        print(f"{name}:  L[ {cells} ]R   -> _continuous / _pulse / _sweep .wav")


if __name__ == "__main__":
    main()
