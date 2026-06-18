"""Real-time echolocation: webcam -> metric depth -> spatial audio.

Walk around with headphones on and compare the three modes live.

Run:
    python3 src/live.py                 # default: continuous mode
    python3 src/live.py --mode sweep --sweep 0.2
    python3 src/live.py --selftest      # no camera/audio; checks the pipeline

Keys (in the preview window):
    1 / 2 / 3   continuous / pulse / sweep
    [ / ]       range (far): nearer / farther cutoff
    - / =        loudness falloff: gentler / steeper
    a / s       SIDE   pitch (outer edges): down / up
    k / l       CENTER pitch (straight ahead): down / up
    , / .       sweep: slower / faster
    q or ESC    quit

macOS: the FIRST run needs Camera permission for your terminal app
(System Settings -> Privacy & Security -> Camera). Use headphones.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS lacks one op
import argparse
import time
import numpy as np
import cv2
import torch
from PIL import Image

from synth import Synth, sector_distances, MODES

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
ROI_TOP, ROI_BOTTOM = 0.10, 0.80
N_SECTORS = 7
SEMI = 2 ** (1 / 12)  # one semitone, for pitch knobs

KEYMAP = (
    "keys: 1/2/3 mode | [ ] range | - = falloff | "
    "a/s side-pitch | k/l center-pitch | , . sweep | q quit"
)


class DepthEngine:
    def __init__(self, size=322, device=None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.proc.size = {"height": size, "width": size}
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
        print(f"depth model on {device}, input {size}px")

    def infer(self, rgb):
        inp = self.proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            d = self.model(**inp).predicted_depth      # (1, h, w), meters
        return d.squeeze().float().cpu().numpy()


def colorize_depth(depth, near, far, h):
    norm = np.clip((depth - near) / (far - near), 0, 1)
    u8 = ((1 - norm) * 255).astype(np.uint8)            # near = hot
    cm = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    return cv2.resize(cm, (int(cm.shape[1] * h / cm.shape[0]), h))


def draw_overlay(frame, dists, syn, fps):
    vis = frame.copy()
    h, w = vis.shape[:2]
    near, far = syn.near_m, syn.far_m
    top, bot = int(ROI_TOP * h), int(ROI_BOTTOM * h)
    edges = np.linspace(0, w, N_SECTORS + 1).astype(int)
    cv2.line(vis, (0, top), (w, top), (255, 255, 255), 1)
    cv2.line(vis, (0, bot), (w, bot), (255, 255, 255), 1)
    for i, d in enumerate(dists):
        x0, x1 = edges[i], edges[i + 1]
        cv2.line(vis, (x0, top), (x0, bot), (180, 180, 180), 1)
        prox = float(np.clip((far - d) / (far - near), 0, 1))
        col = (0, int(255 * (1 - prox)), int(255 * prox))  # BGR: green=far, red=near
        label = f"{d:.1f}" if d < far else "open"
        cv2.putText(vis, label, (x0 + 4, top + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
        bar = int(prox * (bot - top))
        cv2.rectangle(vis, (x0 + 2, bot - bar), (x1 - 2, bot), col, -1)
    # suggested path = most-open sector
    j = int(np.argmax(dists))
    cx = (edges[j] + edges[j + 1]) // 2
    pcol = (0, 255, 0) if dists[j] >= far else (0, 220, 220)
    cv2.arrowedLine(vis, (w // 2, h - 8), (cx, top + 30), pcol, 3, tipLength=0.25)
    hud = (f"{syn.mode}  far {far:.1f}m  fall {syn.falloff:.2f}  "
           f"side {syn.side_hz:.0f}Hz  ctr {syn.center_hz:.0f}Hz  "
           f"sweep {syn.sweep_period:.2f}s  {fps:.0f}fps")
    cv2.putText(vis, hud, (8, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, KEYMAP, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 220, 180), 1, cv2.LINE_AA)
    return vis


def build_synth(args):
    return Synth(n_sectors=N_SECTORS, near_m=0.5, far_m=args.far,
                 side_hz=args.side, center_hz=args.center, falloff=args.falloff,
                 sweep_period=args.sweep, mode=args.mode)


def selftest(args):
    eng = DepthEngine(size=args.size)
    path = args.image or "/Users/levilingsch/Code/AISprint/work/rgb/IMG_0281.png"
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    depth = eng.infer(rgb)
    dists = sector_distances(depth, n_sectors=N_SECTORS, roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM)
    print("depth", depth.shape, "dists", np.round(dists, 2))
    syn = build_synth(args)
    syn.set_distances(dists)
    peak, finite = 0.0, True
    for _ in range(int(2.0 * syn.sr / 512)):  # ~2 s, enough for a full sweep
        blk = syn.render(512)
        peak = max(peak, float(np.abs(blk).max()))
        finite = finite and bool(np.isfinite(blk).all())
    print(f"render 2s peak={peak:.3f} finite={finite}  freqs={np.round(syn.freq).tolist()}")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="continuous", choices=MODES)
    ap.add_argument("--size", type=int, default=322)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--far", type=float, default=3.0)
    ap.add_argument("--side", type=float, default=300.0, help="outer-edge pitch (Hz)")
    ap.add_argument("--center", type=float, default=800.0, help="center/ahead pitch (Hz)")
    ap.add_argument("--falloff", type=float, default=2.0, help="loudness falloff exponent (1=linear, 2=quadratic)")
    ap.add_argument("--sweep", type=float, default=0.3, help="seconds per left->right scan")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image", default=None, help="selftest source image")
    args = ap.parse_args()

    if args.selftest:
        selftest(args)
        return

    import sounddevice as sd
    eng = DepthEngine(size=args.size, device=args.device)
    syn = build_synth(args)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Could not open camera (check macOS Camera permission).")

    def audio_cb(outdata, frames, t, status):
        outdata[:] = syn.render(frames)

    stream = sd.OutputStream(samplerate=syn.sr, channels=2, dtype="float32",
                             blocksize=512, callback=audio_cb)
    stream.start()
    print("running. Put headphones on. Press q to quit.\n" + KEYMAP)

    last, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = eng.infer(rgb)
            dists = sector_distances(depth, n_sectors=N_SECTORS,
                                     roi_top=ROI_TOP, roi_bottom=ROI_BOTTOM)
            syn.set_distances(dists)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            vis = draw_overlay(frame, dists, syn, fps)
            dvis = colorize_depth(depth, syn.near_m, syn.far_m, vis.shape[0])
            cv2.imshow("echolocation", np.hstack([vis, dvis]))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("1"): syn.set_mode("continuous")
            elif k == ord("2"): syn.set_mode("pulse")
            elif k == ord("3"): syn.set_mode("sweep")
            elif k == ord("["): syn.far_m = max(2.0, syn.far_m - 0.5)
            elif k == ord("]"): syn.far_m = min(15.0, syn.far_m + 0.5)
            elif k == ord("-"): syn.falloff = max(0.2, syn.falloff - 0.15)
            elif k == ord("="): syn.falloff = min(4.0, syn.falloff + 0.15)
            elif k == ord("a"): syn.set_freqs(side=syn.side_hz / SEMI)
            elif k == ord("s"): syn.set_freqs(side=syn.side_hz * SEMI)
            elif k == ord("k"): syn.set_freqs(center=syn.center_hz / SEMI)
            elif k == ord("l"): syn.set_freqs(center=syn.center_hz * SEMI)
            elif k == ord(","): syn.sweep_period = min(3.0, syn.sweep_period + 0.05)
            elif k == ord("."): syn.sweep_period = max(0.1, syn.sweep_period - 0.05)
    finally:
        stream.stop(); stream.close()
        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
