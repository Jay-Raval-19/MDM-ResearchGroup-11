"""Closed-form spectral indices and the near-miss decoys used to validate the battery.

Inputs are reflectance x 10,000 with the band axis at position 1, e.g. (N, 6, H, W).
Formulas are evaluated on reflectance fractions (0-1), because SAVI's L = 0.5 and
EVI's constants are defined on that scale.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .bands import BAND_INDEX, REFLECTANCE_SCALE

IndexFn = Callable[[np.ndarray], np.ndarray]


def band(x: np.ndarray, name: str) -> np.ndarray:
    return np.take(x, BAND_INDEX[name], axis=1) / REFLECTANCE_SCALE


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN unless both bands are positive.

    Real HLS surface reflectance has about 0.5% of pixels at or below zero
    (atmospheric-correction artifacts over water and deep terrain shadow). A
    negative band gives impossible values such as NDVI > 1, so those pixels
    have no index value."""
    ok = (a > 0) & (b > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (a - b) / (a + b), np.nan)


def pair_index(a: str, b: str) -> IndexFn:
    """ND(a, b) as a function of a reflectance array."""
    return lambda x: normalized_difference(band(x, a), band(x, b))


# Every pair of the six bands, with the standard name where one exists. The six pre-registered
# indices come first so that v3 columns line up with the earlier runs; MNDWI and the eight unnamed
# pairs follow. Unnamed pairs are labelled by initials (B, G, R, N = NIR, S1, S2).
PAIR_TARGETS: dict[str, tuple[str, str]] = {
    "NDVI": ("nir", "red"),
    "NDWI": ("green", "nir"),
    "NDMI": ("nir", "swir1"),
    "NBR": ("nir", "swir2"),
    "NGRDI": ("green", "red"),
    "NBR2": ("swir1", "swir2"),
    "MNDWI": ("green", "swir1"),
    "B-G": ("blue", "green"),
    "B-R": ("blue", "red"),
    "B-N": ("blue", "nir"),
    "B-S1": ("blue", "swir1"),
    "B-S2": ("blue", "swir2"),
    "G-S2": ("green", "swir2"),
    "R-S1": ("red", "swir1"),
    "R-S2": ("red", "swir2"),
}
NAMED_PAIRS = ("NDVI", "NDWI", "NDMI", "NBR", "NGRDI", "NBR2", "MNDWI")
CORE_PAIRS = ("NDVI", "NDWI", "NDMI", "NBR", "NGRDI", "NBR2")  # the v1/v2 pre-registered set


def pair_table(which: str = "all15") -> dict[str, tuple[IndexFn, str]]:
    """label -> (per-pixel index function, expected battery class). `which`: all15 | core | named."""
    keys = {"all15": tuple(PAIR_TARGETS), "core": CORE_PAIRS, "named": NAMED_PAIRS}[which]
    return {k: (pair_index(*PAIR_TARGETS[k]), f"ND({PAIR_TARGETS[k][0]},{PAIR_TARGETS[k][1]})") for k in keys}


def ndvi(x):
    return normalized_difference(band(x, "nir"), band(x, "red"))


def ndwi(x):  # McFeeters open-water index
    return normalized_difference(band(x, "green"), band(x, "nir"))


def ndmi(x):  # Gao's NIR-SWIR1 index, also called NDWI in some literature
    return normalized_difference(band(x, "nir"), band(x, "swir1"))


def nbr(x):
    return normalized_difference(band(x, "nir"), band(x, "swir2"))


def simple_ratio(x):
    return band(x, "nir") / band(x, "red")


def savi(x, L: float = 0.5):
    n, r = band(x, "nir"), band(x, "red")
    return (1 + L) * (n - r) / (n + r + L)


def evi(x):
    n, r, b = band(x, "nir"), band(x, "red"), band(x, "blue")
    return 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1)


def nir_minus_red(x):
    return band(x, "nir") - band(x, "red")


def nir(x):
    return band(x, "nir")


def brightness(x):
    return np.mean(x, axis=1) / REFLECTANCE_SCALE


def leaky_ndvi(eps: float) -> IndexFn:
    """NDVI plus a small NIR leak: looks like NDVI in direction, but is not degree-zero."""
    return lambda x: ndvi(x) + eps * band(x, "nir")


def patch_mean(index_fn: IndexFn) -> IndexFn:
    """Per-pixel index averaged over each patch: (N, 6, H, W) -> (N,)."""
    return lambda x: index_fn(x).reshape(x.shape[0], -1).mean(axis=1)


def index_of_mean(index_fn: IndexFn) -> IndexFn:
    """Index of the patch's mean spectrum: (N, 6, H, W) -> (N,)."""
    return lambda x: index_fn(x.reshape(x.shape[0], x.shape[1], -1).mean(axis=2))


brightness_patch = patch_mean(brightness)


def near_miss_suite() -> dict[str, tuple[IndexFn, str]]:
    """Patch-level functions with their correct verdicts, for the Stage 0 metric check.

    An expected value of "ratio:<class>" means the verdict must be ratio-type in that
    class; "not-ratio" means any verdict except ratio-type passes; anything else must
    be the best-matching class.
    """
    return {
        "NDVI (patch mean)": (patch_mean(ndvi), "ratio:ND(nir,red)"),
        "NDVI (of mean spectrum)": (index_of_mean(ndvi), "ratio:ND(nir,red)"),
        "Simple ratio NIR/Red": (patch_mean(simple_ratio), "ratio:ND(nir,red)"),
        "NDWI": (patch_mean(ndwi), "ratio:ND(green,nir)"),
        "NDMI": (patch_mean(ndmi), "ratio:ND(nir,swir1)"),
        "NBR": (patch_mean(nbr), "ratio:ND(nir,swir2)"),
        "NIR - Red": (patch_mean(nir_minus_red), "not-ratio"),
        "NIR alone": (patch_mean(nir), "band(nir)"),
        "Brightness": (brightness_patch, "broadband"),
        "SAVI": (patch_mean(savi), "not-ratio"),
        "EVI": (patch_mean(evi), "not-ratio"),
        "NDVI + 0.1*NIR": (patch_mean(leaky_ndvi(0.1)), "not-ratio"),
    }
