"""Band definitions, spectral edits and Prithvi input normalization.

Conventions
-----------
* Reflectance arrays hold HLS surface reflectance x 10,000 (the HLS storage
  convention, and the units Prithvi's normalization constants are in).
* Batched arrays are shaped (N, 6, H, W) with the band axis at position 1.
* Every spectral edit happens in reflectance space, *before* normalize().
  Scaling reflectance by k is not the same as scaling the normalized input.
"""
from __future__ import annotations

import numpy as np

BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")
BAND_INDEX = {name: i for i, name in enumerate(BANDS)}
HLS_L30 = ("B02", "B03", "B04", "B05", "B06", "B07")  # Landsat names, Prithvi config order
HLS_S30 = ("B02", "B03", "B04", "B8A", "B11", "B12")  # Sentinel-2 equivalents
CENTER_NM = (482, 561, 655, 865, 1609, 2201)
REFLECTANCE_SCALE = 10_000.0
BAND_AXIS = 1

# From the Prithvi-EO-2.0 pretrained_cfg (identical for tiny, 300M and 600M).
PRITHVI_MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0])
PRITHVI_STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0])


def _band_shape(ndim: int, axis: int) -> tuple[int, ...]:
    shape = [1] * ndim
    shape[axis] = len(BANDS)
    return tuple(shape)


def scale_vector(scales) -> np.ndarray:
    """Turn {"red": 1.1, "nir": 1.1} or a length-6 sequence into six band factors."""
    if isinstance(scales, dict):
        v = np.ones(len(BANDS))
        for name, k in scales.items():
            v[BAND_INDEX[name]] = k
        return v
    v = np.asarray(scales, dtype=float)
    if v.shape != (len(BANDS),):
        raise ValueError(f"expected 6 band factors, got shape {v.shape}")
    return v


def edit(x: np.ndarray, scales, axis: int = BAND_AXIS) -> np.ndarray:
    """Multiply bands by per-band factors. Returns a new array; x is untouched."""
    return x * scale_vector(scales).reshape(_band_shape(x.ndim, axis))


def mix(x: np.ndarray, endmember, alpha: float, axis: int = BAND_AXIS) -> np.ndarray:
    """Linear spectral mixing toward a pure spectrum: (1 - alpha) * x + alpha * endmember.

    `endmember` is a length-6 spectrum in the same units as x. Mixed pixels are how
    real pixels form, so these edits stay closer to the training distribution than
    large single-band scalings do.
    """
    e = np.asarray(endmember, dtype=float).reshape(_band_shape(x.ndim, axis))
    return (1.0 - alpha) * x + alpha * e


def normalize(x: np.ndarray, axis: int = BAND_AXIS) -> np.ndarray:
    """Prithvi input normalization. Apply only after all edits."""
    shape = _band_shape(x.ndim, axis)
    return (x - PRITHVI_MEAN.reshape(shape)) / PRITHVI_STD.reshape(shape)
