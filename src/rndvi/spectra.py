"""Illustrative endmember spectra and a synthetic patch generator.

These are round-number surface reflectances (x 10,000) for teaching and for the
Stage 0 metric check. Real endmembers come from the USGS Spectral Library in Stage 1.
Band order: blue, green, red, nir, swir1, swir2.
"""
from __future__ import annotations

import numpy as np

ENDMEMBERS = {
    "grass": [300, 700, 400, 4000, 2000, 1000],
    "forest": [200, 400, 250, 2800, 1200, 500],
    "soil": [1000, 1400, 1800, 2500, 3200, 2800],
    "burn": [400, 500, 700, 1200, 2000, 1800],
    "water": [600, 500, 300, 100, 50, 30],
}


def synthetic_patches(n: int = 2000, size: int = 4, seed: int = 0,
                      illumination=(0.6, 1.4), noise: float = 0.01) -> np.ndarray:
    """Mixed-pixel patches: each patch blends two endmembers pixel by pixel.

    Each patch also gets one illumination factor (shading or sun angle), and each
    pixel gets 1% multiplicative sensor noise. Returns an array of shape
    (n, 6, size, size).
    """
    rng = np.random.default_rng(seed)
    E = np.array(list(ENDMEMBERS.values()), dtype=float)
    out = np.empty((n, E.shape[1], size, size))
    for k in range(n):
        pair = rng.choice(len(E), size=2, replace=False)
        weights = rng.dirichlet([0.7, 0.7], size=size * size)
        pixels = weights @ E[pair] * rng.uniform(*illumination)
        pixels *= 1 + noise * rng.standard_normal(pixels.shape)
        out[k] = np.clip(pixels, 1.0, None).T.reshape(E.shape[1], size, size)
    return out
