"""Project paths and provisional thresholds.

Large files (HLS tiles, checkpoints, activations) go to DATA_ROOT, which defaults
to a folder outside OneDrive so tens of gigabytes don't get synced.
Override with the RNDVI_DATA environment variable.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("RNDVI_DATA", Path.home() / "rndvi-data"))
RUNS_DIR = REPO_ROOT / "runs"

# Optional cap on this process's GPU memory (fraction of the card), e.g. RNDVI_GPU_FRACTION=0.85 on a laptop:
# an oversized job then fails with an out-of-memory error instead of spilling into system RAM, which can
# freeze a Windows machine whose display shares the GPU.
if os.environ.get("RNDVI_GPU_FRACTION"):
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ["RNDVI_GPU_FRACTION"]))

# PROVISIONAL. Calibrated on synthetic models in Stage 0.5, then frozen in
# notes/prereg/thresholds.md. Do not tune these on Prithvi results.
THRESHOLDS = {
    "direction": 0.90,  # min |median cosine| to the best class template
    "ratio_residual": 0.05,  # max |sum e| / sum |e| for a ratio-type (degree-zero) verdict
    "margin": 0.10,  # min lower-CI margin between best and runner-up class
    "magnitude": 0.10,  # min median ||e|| / std(f): excludes aligned-but-dead features
}
