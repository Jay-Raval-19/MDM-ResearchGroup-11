"""Unit tests for bands, indices and the battery. Run: python tests/run_tests.py (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rndvi import indices as ix  # noqa: E402
from rndvi.bands import BAND_INDEX, edit, mix, normalize  # noqa: E402
from rndvi.battery import all_band_invariance, elasticities, pair_invariance, PAIRS  # noqa: E402
from rndvi.spectra import ENDMEMBERS, synthetic_patches  # noqa: E402

GRASS = np.array(ENDMEMBERS["grass"], float).reshape(1, 6)
SOIL = np.array(ENDMEMBERS["soil"], float).reshape(1, 6)
WATER = np.array(ENDMEMBERS["water"], float).reshape(1, 6)


def _pair(a, b):
    return next(p for p, (i, j) in enumerate(PAIRS) if {i, j} == {BAND_INDEX[a], BAND_INDEX[b]})


# ---- bands

def test_edit_identity_and_copy():
    x = synthetic_patches(5, seed=1)
    y = edit(x, np.ones(6))
    assert np.array_equal(x, y) and y is not x


def test_edit_happens_before_normalize():
    x = synthetic_patches(5, seed=2)
    right = normalize(edit(x, {"nir": 1.1}))
    wrong = edit(normalize(x), {"nir": 1.1})
    assert not np.allclose(right, wrong), "scaling normalized input must differ from scaling reflectance"


def test_mix_endpoints():
    x = synthetic_patches(3, seed=3)
    s = ENDMEMBERS["forest"]
    assert np.allclose(mix(x, s, 0.0), x)
    assert np.allclose(mix(x, s, 1.0), np.broadcast_to(np.array(s, float).reshape(1, 6, 1, 1), x.shape))


# ---- indices (values from the primer table)

def test_primer_values():
    assert np.isclose(ix.ndvi(GRASS)[0], 0.36 / 0.44)
    assert np.isclose(ix.ndvi(GRASS * 0.5)[0], ix.ndvi(GRASS)[0])  # grass in shadow
    assert np.isclose(ix.ndvi(SOIL)[0], 0.07 / 0.43)
    assert np.isclose(ix.ndvi(WATER)[0], -0.5)
    assert np.isclose(ix.simple_ratio(GRASS)[0], 10.0)


def test_savi_not_brightness_invariant():
    assert not np.isclose(ix.savi(GRASS * 0.5)[0], ix.savi(GRASS)[0])


# ---- battery

def test_ndvi_elasticities_cancel():
    x = synthetic_patches(50, seed=4)
    e = elasticities(ix.patch_mean(ix.ndvi), x)[:, 0, :]
    assert np.allclose(e.sum(-1), 0, atol=1e-6)
    assert np.allclose(e[:, BAND_INDEX["red"]], -e[:, BAND_INDEX["nir"]], atol=1e-6)
    others = [BAND_INDEX[b] for b in ("blue", "green", "swir1", "swir2")]
    assert np.allclose(e[:, others], 0, atol=1e-9)


def test_brightness_elasticities_positive():
    x = synthetic_patches(50, seed=5)
    assert (elasticities(ix.brightness_patch, x) > 0).all()


def test_pair_invariance_identifies_ndvi_pair():
    x = synthetic_patches(200, seed=6)
    S = pair_invariance(ix.patch_mean(ix.ndvi), x)[0]
    assert np.isclose(S[_pair("red", "nir")], 1.0, atol=1e-9)
    assert abs(S[_pair("green", "nir")]) < 0.05  # behaves like scaling NIR alone
    assert np.isnan(S[_pair("blue", "green")])  # responds to neither band


def test_all_band_invariance_separates_families():
    x = synthetic_patches(200, seed=7)
    assert np.isclose(all_band_invariance(ix.patch_mean(ix.ndvi), x)[0], 1.0, atol=1e-9)
    assert all_band_invariance(ix.patch_mean(ix.nir_minus_red), x)[0] < 0.9
    assert all_band_invariance(ix.brightness_patch, x)[0] < 0.5
