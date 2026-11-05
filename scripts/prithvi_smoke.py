"""Step 7 smoke test: does Prithvi-EO-2.0-tiny-TL run, and does our wrapper behave?

1. Runs the official inference.py demo on the four example tiles. This is the vendor's
   own script, run as a subprocess and reported rather than asserted: on torch 2.11 it
   exits 1 at its `torch.Tensor(..., device=device)` legacy constructor, which newer
   torch rejects for cuda. That is a bug in the shipped demo, not in this repo, and the
   summary records the exit code and traceback instead of hiding or fixing it.
2. Loads the model single-date through rndvi.prithvi and checks shapes, that a
   random-init copy differs, that an identity splice changes nothing, and that
   single-date masked reconstruction works far better than random init. These are the
   checks that gate the rest of the project.
Writes runs/step07_smoke/summary.md.

    C:\\Users\\jayra\\venvs\\rndvi\\Scripts\\python scripts/prithvi_smoke.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rndvi.config import RUNS_DIR  # noqa: E402
from rndvi.prithvi import MODEL_DIRS, load_prithvi, model_inputs, splice, token_activations  # noqa: E402

OUT = RUNS_DIR / "step07_smoke"
MODEL_DIR = MODEL_DIRS["tiny-tl"]


def read_example(path: Path, size: int = 224):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float64)
        lnglat = src.lnglat()
    h, w = img.shape[1:]
    top, left = (h - size) // 2, (w - size) // 2
    tile = img[:, top:top + size, left:left + size]
    stamp = re.search(r"(\d{7})T\d{6}", path.name).group(1)
    return tile, (int(stamp[:4]), int(stamp[4:])), lnglat, img.shape


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Step 7 smoke test: Prithvi-EO-2.0-tiny-TL", ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines.append(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), "
                 f"torch {torch.__version__}")

    # 1. Official demo
    examples = sorted((MODEL_DIR / "examples").glob("*.tif"))
    demo_out = OUT / "official_demo"
    proc = subprocess.run([sys.executable, "inference.py", "--data_files", *map(str, examples),
                           "--output_dir", str(demo_out)], cwd=MODEL_DIR, capture_output=True, text=True)
    produced = sorted(p.name for p in demo_out.glob("*")) if demo_out.exists() else []
    lines += ["", "## Official inference.py (4 dates)", "",
              f"Exit code {proc.returncode}; {len(produced)} output files: {', '.join(produced[:8])}"]
    if proc.returncode != 0:
        lines.append(f"```\n{proc.stderr[-2000:]}\n```")

    # 2. Our wrapper, single date
    tile, date_, lnglat, full_shape = read_example(examples[0])
    lines += ["", "## Our wrapper (single date)", "",
              f"Example {examples[0].name}: full raster {full_shape}, centre crop {tile.shape}; "
              f"reflectance range {tile.min():.0f} to {tile.max():.0f}; "
              f"no-data (-9999) pixels: {int((tile == -9999).sum())}; date {date_}; lnglat {lnglat}."]
    x = tile[None]
    dates, locs = np.array([date_]), np.array([lnglat])
    layers = list(range(12))

    trained = load_prithvi("tiny-tl", num_frames=1, device=device)
    rand = load_prithvi("tiny-tl", num_frames=1, random_init_seed=0, device=device)
    acts = token_activations(trained, x, layers, dates, locs)
    acts_r = token_activations(rand, x, layers, dates, locs)
    lines.append(f"Token activations per layer: {acts[5].shape} (tiles, tokens, width); {len(acts)} layers.")
    rel = [np.linalg.norm(acts[i] - acts_r[i]) / np.linalg.norm(acts[i]) for i in layers]
    lines.append(f"Trained vs random-init, relative difference by layer: {np.round(rel, 2).tolist()}")

    with splice(trained, 5, lambda out: out):
        same = token_activations(trained, x, [11], dates, locs)[11]
    with splice(trained, 5, lambda out: torch.zeros_like(out)):
        zeroed = token_activations(trained, x, [11], dates, locs)[11]
    lines.append(f"Identity splice at layer 5 leaves layer 11 unchanged: {np.allclose(same, acts[11], atol=1e-5)}; "
                 f"zeroing layer 5 changes it: {not np.allclose(zeroed, acts[11], atol=1e-3)}")

    # Single-date masked reconstruction: trained should beat random init by a wide margin
    xt, t, loc = model_inputs(x, trained.encoder.patch_embed.proj.weight.device, dates, locs)
    losses = {}
    with torch.no_grad():
        for name, m in [("trained", trained), ("random-init", rand)]:
            torch.manual_seed(0)
            losses[name] = float(m(xt, t, loc, mask_ratio=0.75)[0])
    lines.append(f"Masked-reconstruction loss at 75% masking, single date (normalized units): "
                 f"trained {losses['trained']:.3f}, random-init {losses['random-init']:.3f}")

    # Preview of step 32: shift the date by half a year with pixels fixed
    shifted = token_activations(trained, x, [11], np.array([[date_[0], (date_[1] + 182 - 1) % 365 + 1]]), locs)[11]
    lines.append(f"Preview: day-of-year +182, pixels fixed; layer-11 relative change "
                 f"{np.linalg.norm(shifted - acts[11]) / np.linalg.norm(acts[11]):.3f}")

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
