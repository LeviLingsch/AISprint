"""Metric (absolute, meters) depth estimation for the echolocation prototype.

Uses Depth Anything V2 Metric (Indoor) — outputs depth in METERS, which is what
we need for distance-based sonification (not relative/normalized depth).

Run once to populate output/depth/*.npy; sonify.py then iterates off those.
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Small = fast (~100MB), Base/Large = better. Override with DEPTH_MODEL env var.
MODEL_ID = os.environ.get(
    "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
)
ROOT = "/Users/levilingsch/Code/AISprint"
RGB_DIR = os.path.join(ROOT, "work/rgb")
OUT_DIR = os.path.join(ROOT, "output/depth")


def estimate_depth(image, proc, model, device):
    """Return a metric depth map (H, W) in meters at the image's resolution."""
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth  # (1, h, w), meters
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return pred


def main():
    device = "cpu"  # 4 images, fine on CPU; switch to mps/cuda for batch later
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading {MODEL_ID} on {device} ...")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()

    for p in sorted(glob.glob(os.path.join(RGB_DIR, "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        image = Image.open(p).convert("RGB")
        depth = estimate_depth(image, proc, model, device)
        np.save(os.path.join(OUT_DIR, name + ".npy"), depth.astype(np.float32))
        print(
            f"{name}: {depth.shape}  near={depth.min():.2f}m  "
            f"median={np.median(depth):.2f}m  far={depth.max():.2f}m"
        )

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].imshow(image); ax[0].set_title("RGB"); ax[0].axis("off")
        im = ax[1].imshow(depth, cmap="turbo")
        ax[1].set_title("Depth (meters)"); ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, name + "_depth.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
