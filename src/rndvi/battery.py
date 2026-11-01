"""The spectral intervention battery and its scoring.

A *feature function* maps reflectance arrays (N, 6, H, W) to values shaped (N,) or
(N, F). Today that's a closed-form index; later it's "run Prithvi on the edited
tile and read SAE latent i at token t". The battery only ever calls it on edited
copies of the input, so the same code tests formulas and neural features.

Mechanism classes
-----------------
* ND(a,b): a normalized difference (x_a - x_b)/(x_a + x_b), or any monotone
  function of x_a/x_b. Its elasticity vector points along +a -b at every pixel.
  Each of the 15 band pairs is one class, and four are pre-registered.
* band(b): a single-band response.
* broadband: a positive-weighted band sum, whose elasticities are proportional
  to the spectrum itself.
Classes are compared by their elasticity direction. Ratio-type (degree-zero)
behaviour is checked separately, with the ratio residual.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from .bands import BANDS, BAND_INDEX, edit

FeatureFn = Callable[[np.ndarray], np.ndarray]
PAIRS = list(combinations(range(len(BANDS)), 2))  # 15 unordered band pairs
PREREGISTERED = {  # class name -> (positive band, negative band, family)
    "ND(nir,red)": ("nir", "red", "NDVI family"),
    "ND(green,nir)": ("green", "nir", "NDWI family"),
    "ND(nir,swir1)": ("nir", "swir1", "NDMI family"),
    "ND(nir,swir2)": ("nir", "swir2", "NBR family"),
}
_TINY = 1e-12


def _as_2d(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v[:, None] if v.ndim == 1 else v


def _pair_template(pos: str, neg: str) -> np.ndarray:
    t = np.zeros(len(BANDS))
    t[BAND_INDEX[pos]], t[BAND_INDEX[neg]] = 1.0, -1.0
    return t / np.sqrt(2.0)


def class_templates() -> dict[str, np.ndarray]:
    """Fixed elasticity-direction templates: 15 ND pairs and 6 single bands."""
    templates = {name: _pair_template(pos, neg) for name, (pos, neg, _) in PREREGISTERED.items()}
    registered = {frozenset((BAND_INDEX[p], BAND_INDEX[n])) for p, n, _ in PREREGISTERED.values()}
    for i, j in PAIRS:
        if frozenset((i, j)) not in registered:
            templates[f"ND({BANDS[i]},{BANDS[j]})"] = _pair_template(BANDS[i], BANDS[j])
    for b in BANDS:
        t = np.zeros(len(BANDS))
        t[BAND_INDEX[b]] = 1.0
        templates[f"band({b})"] = t
    return templates


def _pair_index(class_name: str) -> int | None:
    if not class_name.startswith("ND("):
        return None
    wanted = set(class_name[3:-1].split(","))
    return next(p for p, (i, j) in enumerate(PAIRS) if {BANDS[i], BANDS[j]} == wanted)


# ---------------------------------------------------------------- measurements

def elasticities(f: FeatureFn, x: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Log-derivatives e_b = d f / d log x_b by central differences, one band at a time.

    Steps are symmetric in log space (x_b * e^+eps and x_b * e^-eps). Asymmetric
    steps such as x1.05 / x0.95 leave a spurious ratio residual of about 5e-4 on
    NDVI (found by the Stage 0 metric check). With symmetric steps, any two-band
    ratio cancels exactly. Returns (N, F, 6). Any ratio-type computation has
    sum_b e_b = 0 (Euler).
    """
    up_k, down_k = np.exp(eps), np.exp(-eps)
    cols = []
    for b in range(len(BANDS)):
        up, down = np.ones(len(BANDS)), np.ones(len(BANDS))
        up[b], down[b] = up_k, down_k
        cols.append((_as_2d(f(edit(x, up))) - _as_2d(f(edit(x, down)))) / (2 * eps))
    return np.stack(cols, axis=-1)


def _single_band_changes(f: FeatureFn, x: np.ndarray, base: np.ndarray, k: float) -> np.ndarray:
    """|f(x with band b scaled by k) - f(x)| for each band: (N, F, 6)."""
    out = []
    for b in range(len(BANDS)):
        v = np.ones(len(BANDS))
        v[b] = k
        out.append(np.abs(_as_2d(f(edit(x, v))) - base))
    return np.stack(out, axis=-1)


def pair_invariance(f: FeatureFn, x: np.ndarray, k: float = 1.1) -> np.ndarray:
    """S_ij = 1 - median|change when bands i and j are scaled together|
                / max(median|change when i alone is scaled|, same for j).

    1 means invariant to that pair (the feature reads the ratio x_i/x_j); about 0
    means it responds as if one band had been scaled; NaN means it responds to
    neither band. Returns (F, 15) in PAIRS order.
    """
    base = _as_2d(f(x))
    single = np.median(_single_band_changes(f, x, base, k), axis=0)  # (F, 6)
    S = np.full((base.shape[1], len(PAIRS)), np.nan)
    for p, (i, j) in enumerate(PAIRS):
        v = np.ones(len(BANDS))
        v[i] = v[j] = k
        change = np.median(np.abs(_as_2d(f(edit(x, v))) - base), axis=0)
        ref = np.maximum(single[:, i], single[:, j])
        S[:, p] = np.where(ref > _TINY, 1.0 - change / np.maximum(ref, _TINY), np.nan)
    return S


def all_band_invariance(f: FeatureFn, x: np.ndarray, k: float = 1.1) -> np.ndarray:
    """1 - median|change when every band is scaled by k| / median L2 norm of the six
    single-band changes. 1 = brightness-invariant (ratio family). Returns (F,)."""
    base = _as_2d(f(x))
    ref = np.median(np.linalg.norm(_single_band_changes(f, x, base, k), axis=-1), axis=0)
    change = np.median(np.abs(_as_2d(f(edit(x, np.full(len(BANDS), k)))) - base), axis=0)
    return np.where(ref > _TINY, 1.0 - change / np.maximum(ref, _TINY), np.nan)


# ---------------------------------------------------------------- scoring

def sample_spectra(x: np.ndarray) -> np.ndarray:
    """Mean spectrum per sample: (N, 6, ...) -> (N, 6)."""
    return x.reshape(x.shape[0], x.shape[1], -1).mean(axis=2)


def alignment_cosines(e: np.ndarray, x: np.ndarray | None = None,
                      spectra: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    """Per-sample signed cosine between each feature's elasticity vector and every
    class template, including the spectrum-dependent broadband template.

    Pass `spectra` (N, 6) when samples aren't the rows of x, e.g. one sample per
    Prithvi token, where x holds whole tiles. Returns (names, array (N, F, C)).
    """
    templates = class_templates()
    names = list(templates) + ["broadband"]
    T = np.stack(list(templates.values()))
    e_unit = e / (np.linalg.norm(e, axis=-1, keepdims=True) + _TINY)
    cos_fixed = np.einsum("nfb,cb->nfc", e_unit, T)
    spectrum = sample_spectra(x) if spectra is None else spectra
    s_unit = spectrum / (np.linalg.norm(spectrum, axis=-1, keepdims=True) + _TINY)
    cos_broad = np.einsum("nfb,nb->nf", e_unit, s_unit)[..., None]
    return names, np.concatenate([cos_fixed, cos_broad], axis=-1)


def ratio_residual(e: np.ndarray) -> np.ndarray:
    """Median over samples of |sum_b e_b| / sum_b |e_b|: 0 for ratio-type, 1 for single-signed. (F,)"""
    return np.median(np.abs(e.sum(-1)) / (np.abs(e).sum(-1) + _TINY), axis=0)


def magnitude(e: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Median elasticity norm relative to the feature's natural spread. Near 0 = aligned but dead. (F,)"""
    spread = np.std(_as_2d(values), axis=0)
    return np.median(np.linalg.norm(e, axis=-1), axis=0) / np.maximum(spread, _TINY)


def site_bootstrap(f: FeatureFn, x: np.ndarray, groups: np.ndarray, eps: float = 0.05,
                   n_boot: int = 300, seed: int = 0) -> list[dict]:
    """Residual confidence interval from resampling *sites*, not tokens.

    Tokens from one site are highly correlated -- same sensor pass, same land cover -- so a
    token-level bootstrap understates uncertainty. This resamples whole sites with replacement
    (a cluster bootstrap), which is the interval that answers "would another five sites give the
    same verdict?". `groups` labels each row of the feature output with its site.
    """
    e = elasticities(f, x, eps)                                    # (S, F, 6)
    per_token = np.abs(e.sum(-1)) / (np.abs(e).sum(-1) + 1e-12)    # (S, F)
    labels = np.asarray(groups)
    uniq = np.unique(labels)
    idx_of = {g: np.flatnonzero(labels == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, per_token.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_of[g] for g in pick])
        boot[b] = np.median(per_token[rows], axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    per_site = {g: np.median(per_token[i], axis=0) for g, i in idx_of.items()}
    return [{"feature": fi, "residual": float(np.median(per_token[:, fi])),
             "site_ci_lo": float(lo[fi]), "site_ci_hi": float(hi[fi]),
             "worst_site": float(max(v[fi] for v in per_site.values())),
             "best_site": float(min(v[fi] for v in per_site.values())),
             "n_sites": int(len(uniq))} for fi in range(per_token.shape[1])]


def eps_sweep(f: FeatureFn, x: np.ndarray, eps_list=(0.01, 0.02, 0.05, 0.1, 0.2),
              spectra: np.ndarray | None = None) -> list[dict]:
    """Residual and best-class direction at several perturbation sizes, one row per feature.

    A mechanism should look the same at every scale; a verdict that holds only at one eps is a
    local-linearization artifact. Costs 12 forward passes per eps and is shared across all features
    that f returns, so sweeping is cheap relative to fitting.
    """
    out = []
    for eps in eps_list:
        e = elasticities(f, x, eps)
        names, cos = alignment_cosines(e, x, spectra)
        M = np.abs(np.median(cos, axis=0))                     # (F, C)
        r = ratio_residual(e)
        for fi in range(e.shape[1]):
            order = np.argsort(-M[fi])
            out.append({"eps": eps, "feature": fi, "residual": float(r[fi]),
                        "best": names[order[0]], "direction": float(M[fi, order[0]]),
                        "margin": float(M[fi, order[0]] - M[fi, order[1]])})
    return out


def score(f: FeatureFn, x: np.ndarray, thresholds: dict, eps: float = 0.05, k: float = 1.1,
          n_boot: int = 200, seed: int = 0, spectra: np.ndarray | None = None) -> list[dict]:
    """Run the battery on feature function f over inputs x; one verdict per feature.

    Direction, magnitude, ratio residual and class margin are reported separately,
    never multiplied into one number, so it's always visible which criterion failed.
    `spectra` gives one spectrum per output row when f returns several samples per
    input (see alignment_cosines).
    """
    values = _as_2d(f(x))
    e = elasticities(f, x, eps)
    names, cos = alignment_cosines(e, x, spectra)
    M = np.median(cos, axis=0)  # (F, C), signed
    r, mag = ratio_residual(e), magnitude(e, values)
    S_pair, S_all = pair_invariance(f, x, k), all_band_invariance(f, x, k)
    rng = np.random.default_rng(seed)
    results = []
    for fi in range(values.shape[1]):
        order = np.argsort(-np.abs(M[fi]))
        best, second = order[0], order[1]
        boot = np.empty(n_boot)
        for t in range(n_boot):
            idx = rng.integers(0, cos.shape[0], cos.shape[0])
            Mb = np.abs(np.median(cos[idx, fi], axis=0))
            boot[t] = Mb[best] - Mb[second]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        best_name = names[best]
        pair = _pair_index(best_name)
        strong = abs(M[fi, best]) >= thresholds["direction"] and mag[fi] >= thresholds["magnitude"]
        if pair is not None and r[fi] >= thresholds["ratio_residual"]:
            verdict = "near-ratio: fails ratio residual (not ratio-type)"
        elif pair is not None and strong and lo > thresholds["margin"]:
            family = PREREGISTERED.get(best_name, (None, None, "exploratory pair"))[2]
            verdict = f"ratio-type {best_name} [{family}]"
        elif pair is not None:
            verdict = f"ratio-like {best_name}, unresolved"
        else:
            verdict = best_name if strong else f"{best_name}, weak"
        results.append({
            "best": best_name,
            "polarity": "+" if M[fi, best] >= 0 else "-",
            "direction": float(abs(M[fi, best])),
            "runner_up": names[second],
            "margin": float(abs(M[fi, best]) - abs(M[fi, second])),
            "margin_ci": (float(lo), float(hi)),
            "ratio_residual": float(r[fi]),
            "magnitude": float(mag[fi]),
            "all_band_S": float(S_all[fi]),
            "own_pair_S": float(S_pair[fi, pair]) if pair is not None else float("nan"),
            "verdict": verdict,
        })
    return results
