"""Step 13 pilot: do Prithvi's units show ratio-type spectral fingerprints beyond what
untrained copies of the same architecture show?

For each model (trained, plus random-init copies), every tile is run with each of the
12 single-band edits (x e^+-0.05, in reflectance space). Activations at the chosen
layers are collected for a fixed sample of valid tokens, and each token gets an
elasticity vector for every neuron and for the top 50 principal directions of each
layer. Units are scored with the battery's criteria (best class, direction, ratio
residual, magnitude, margin) against the PROVISIONAL thresholds.

    python scripts/pilot.py                                   # tiny-TL, all 12 layers, all tiles
    python scripts/pilot.py --variant 300m-tl --layers 0 3 6 9 12 15 18 21 23 --max-tiles 500
    python scripts/pilot.py --out runs/step13_pilot --report-only

Writes <out>/{summary.md, counts.csv, counts.png, units_<model>.npz}.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")  # the Windows console default (cp1252) can't print ≥ or ±

from rndvi.bands import BANDS, edit, normalize  # noqa: E402
from rndvi.battery import alignment_cosines, magnitude, ratio_residual  # noqa: E402
from rndvi.config import DATA_ROOT, RUNS_DIR, THRESHOLDS  # noqa: E402
from rndvi.prithvi import load_prithvi, patch_spectra, token_activations, valid_patches  # noqa: E402

EPS = 0.05
N_PCS = 50
LOOSE = {"direction": 0.80, "ratio_residual": 0.15}
UNIT_CHUNK = 256  # score units in chunks to bound memory on wide models


def sample_tokens(valid: np.ndarray, k: int, rng) -> np.ndarray:
    """Up to k random valid token indices per tile, as a 1-D object array of int arrays.

    Built element by element: np.array(list_of_arrays, dtype=object) turns into a 2-D
    object array when every tile has exactly k tokens, which breaks fancy indexing."""
    out = np.empty(len(valid), dtype=object)
    for i, v in enumerate(valid):
        idx = np.flatnonzero(v)
        out[i] = rng.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else np.array([], dtype=int)
    return out


def load_tiles(tile_dir: Path, max_tiles: int | None):
    rows = list(csv.DictReader(open(tile_dir / "manifest.csv")))[:max_tiles]
    tiles = np.stack([rasterio.open(tile_dir / r["file"]).read().astype(np.float32) for r in rows])
    dates = np.array([[int(r["year"]), int(r["doy"])] for r in rows], dtype=float)
    locs = np.array([[float(r["lon"]), float(r["lat"])] for r in rows])  # inference.py order (lon, lat)
    return tiles, dates, locs


class LazyTiles:
    """Tiles read from disk on demand, for datasets too large to hold in memory. Supports the indexing the
    scripts use (tiles[i], tiles[slice], tiles[index array]) and returns float32 arrays like load_tiles."""

    def __init__(self, files):
        self.files = list(files)
        self.shape = (len(self.files), 6, 224, 224)

    def __len__(self):
        return len(self.files)

    def _read(self, f):
        with rasterio.open(f) as src:
            return src.read().astype(np.float32)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self._read(self.files[idx])
        return np.stack([self._read(self.files[i]) for i in np.arange(len(self.files))[idx]])


def load_sets(names, lazy: bool | None = None):
    """load_tiles for one or more datasets under DATA_ROOT/hls (e.g. pilot, y2022, offseason), concatenated,
    plus their manifest rows, each tagged with its dataset name in r["set"]. Above 4,000 tiles (or with
    lazy=True) tiles are read from disk on demand instead of loaded at once."""
    rows = []
    for n in names:
        rows += [{**r, "set": n} for r in csv.DictReader(open(DATA_ROOT / "hls" / n / "manifest.csv"))]
    if lazy or (lazy is None and len(rows) > 4000):
        dates = np.array([[int(r["year"]), int(r["doy"])] for r in rows], dtype=float)
        locs = np.array([[float(r["lon"]), float(r["lat"])] for r in rows])
        return LazyTiles(DATA_ROOT / "hls" / r["set"] / r["file"] for r in rows), dates, locs, rows
    parts = [load_tiles(DATA_ROOT / "hls" / n, None) for n in names]
    if len(parts) == 1:
        return (*parts[0], rows)
    return (*(np.concatenate([p[k] for p in parts]) for k in range(3)), rows)


def collect(model, tiles, dates, locs, token_idx, layers, batch, normalizer=normalize):
    """Base activations (S, U) and elasticities (S, U, 6) per layer, for the sampled tokens."""
    base = {l: [] for l in layers}
    elas = {l: [] for l in layers}
    for s in range(0, len(tiles), batch):
        sl = slice(s, s + batch)
        x, d, loc = tiles[sl].astype(np.float64), dates[sl], locs[sl]
        sel = [(i, t) for i, toks in enumerate(token_idx[sl]) for t in toks]
        ti, tt = np.array([a for a, _ in sel]), np.array([b for _, b in sel])
        acts0 = token_activations(model, x, layers, d, loc, batch, normalizer)
        for l in layers:
            base[l].append(acts0[l][ti, tt])
        cols = {l: [] for l in layers}
        for b in range(len(BANDS)):
            up, down = np.ones(6), np.ones(6)
            up[b], down[b] = np.exp(EPS), np.exp(-EPS)
            a_up = token_activations(model, edit(x, up), layers, d, loc, batch, normalizer)
            a_dn = token_activations(model, edit(x, down), layers, d, loc, batch, normalizer)
            for l in layers:
                cols[l].append((a_up[l][ti, tt] - a_dn[l][ti, tt]) / (2 * EPS))
        for l in layers:
            elas[l].append(np.stack(cols[l], axis=-1).astype(np.float32))
    return ({l: np.concatenate(base[l]) for l in layers},
            {l: np.concatenate(elas[l]) for l in layers})


def score_units(e, values, spectra):
    """Per-unit verdict fields from elasticities e (S, U, 6) and activations (S, U)."""
    parts = []
    for u0 in range(0, e.shape[1], UNIT_CHUNK):
        names, cos = alignment_cosines(e[:, u0:u0 + UNIT_CHUNK], spectra=spectra)
        parts.append(np.abs(np.median(cos, axis=0)))
    M = np.concatenate(parts)  # (U, C)
    order = np.argsort(-M, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(M.shape[0])
    direction, margin = M[rows, best], M[rows, best] - M[rows, second]
    r, mag = ratio_residual(e), magnitude(e, values)
    best_names = np.array(names)[best].astype(str)
    is_pair = np.char.startswith(best_names, "ND(")
    strict = (is_pair & (direction >= THRESHOLDS["direction"]) & (r < THRESHOLDS["ratio_residual"])
              & (mag >= THRESHOLDS["magnitude"]) & (margin >= THRESHOLDS["margin"]))
    loose = (is_pair & (direction >= LOOSE["direction"]) & (r < LOOSE["ratio_residual"])
             & (mag >= THRESHOLDS["magnitude"]))
    return dict(best=best_names, direction=direction, margin=margin, residual=r, magnitude=mag,
                strict=strict, loose=loose)


def class_kind(name: str) -> str:
    return "pair" if name.startswith("ND(") else ("broadband" if name == "broadband" else "band")


def table_row(model: str, layer: int, basis: str, sc: dict) -> dict:
    kinds = Counter(class_kind(b) for b in sc["best"])
    return {
        "model": model, "layer": layer, "basis": basis, "units": len(sc["best"]),
        "ratio_strict": int(sc["strict"].sum()), "ratio_loose": int(sc["loose"].sum()),
        "strict_by_pair": json.dumps(dict(Counter(map(str, sc["best"][sc["strict"]])))),
        "best_pair": kinds["pair"], "best_band": kinds["band"], "best_broadband": kinds["broadband"],
    }


def rows_from_saved(model: str, path: Path, layers) -> list[dict]:
    """Rebuild a finished model's table rows from its units_<model>.npz (used when resuming)."""
    s = np.load(path)
    return [table_row(model, l, basis, {k: s[f"L{l}_{basis}_{k}"] for k in ("best", "strict", "loose")})
            for l in layers for basis in ("neurons", "pcs")]


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tiles, dates, locs = load_tiles(DATA_ROOT / "hls" / "pilot", args.max_tiles)
    rng = np.random.default_rng(0)
    valid = valid_patches(tiles).reshape(len(tiles), -1)
    token_idx = sample_tokens(valid, args.tokens_per_tile, rng)
    spectra_all = patch_spectra(tiles.astype(np.float64)).reshape(len(tiles), -1, 6)
    spectra = np.concatenate([spectra_all[i, t] for i, t in enumerate(token_idx)])
    meta = {"variant": args.variant, "tiles": len(tiles), "tokens": len(spectra), "layers": args.layers}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"{args.variant}: {len(tiles)} tiles, {len(spectra)} sampled valid tokens, layers {args.layers}", flush=True)

    models = [("trained", None)] + [(f"random{i}", i) for i in range(args.n_random)]
    table = []
    for name, seed in models:
        done_path = out / f"units_{name}.npz"
        if done_path.exists():  # resume: this model finished in an earlier run
            table += rows_from_saved(name, done_path, args.layers)
            print(f"  {name}: already done, loaded saved results", flush=True)
            continue
        t0 = time.time()
        model = load_prithvi(args.variant, num_frames=1, random_init_seed=seed)
        base, elas = collect(model, tiles, dates, locs, token_idx, args.layers, args.batch)
        del model
        torch.cuda.empty_cache()
        saved = {}
        for l in args.layers:
            centered = base[l] - base[l].mean(0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            P = vt[:N_PCS].T.astype(np.float32)
            for basis, e, vals in [("neurons", elas[l], base[l]),
                                   ("pcs", np.einsum("sub,up->spb", elas[l], P), centered @ P)]:
                sc = score_units(e, vals, spectra)
                table.append(table_row(name, l, basis, sc))
                for k, v in sc.items():
                    saved[f"L{l}_{basis}_{k}"] = v
        tmp = out / f"units_{name}.tmp.npz"
        np.savez_compressed(tmp, **saved)
        tmp.replace(done_path)  # atomic: a half-written file never counts as "done"
        del base, elas
        print(f"  {name}: done in {time.time() - t0:.0f} s", flush=True)

    with open(out / "counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    report(out)


# ---------------------------------------------------------------- reporting

def report(out: Path):
    meta = json.loads((out / "meta.json").read_text())
    layers = meta["layers"]
    table = [{k: (v if k in ("model", "basis", "strict_by_pair") else int(v)) for k, v in r.items()}
             for r in csv.DictReader(open(out / "counts.csv"))]
    randoms = sorted({r["model"] for r in table if r["model"].startswith("random")})
    saved = {m: np.load(out / f"units_{m}.npz") for m in ["trained"] + randoms}

    def get(model, basis, key):
        return np.array([r[key] for r in table if r["model"] == model and r["basis"] == basis])

    lines = [f"# Step 13 pilot · {meta['variant']} vs {len(randoms)} random-init copies", "",
             f"{meta['tiles']} HLS tiles, {meta['tokens']} sampled valid tokens, 12 single-band edits (x e^±{EPS}), "
             f"layers {layers}; eps {EPS}. Strict = provisional thresholds {THRESHOLDS}; loose = direction ≥ "
             f"{LOOSE['direction']}, residual < {LOOSE['ratio_residual']}.", ""]
    verdict_bits = []
    for basis in ("neurons", "pcs"):
        ts, tl = get("trained", basis, "ratio_strict"), get("trained", basis, "ratio_loose")
        rs = np.stack([get(m, basis, "ratio_strict") for m in randoms])
        rl = np.stack([get(m, basis, "ratio_loose") for m in randoms])
        lines += [f"## {basis}", "",
                  "| Layer | Trained strict | Random strict | Trained loose | Random loose | Top trained loose pairs |",
                  "|---|---|---|---|---|---|"]
        for i, l in enumerate(layers):
            b = saved["trained"][f"L{l}_{basis}_best"]
            lo = saved["trained"][f"L{l}_{basis}_loose"]
            top = ", ".join(f"{k[3:-1]}:{v}" for k, v in Counter(b[lo]).most_common(3))
            lines.append(f"| {l} | {ts[i]} | {'/'.join(map(str, rs[:, i]))} | {tl[i]} | "
                         f"{'/'.join(map(str, rl[:, i]))} | {top} |")
        excess = ts - rs.max(0)
        verdict_bits.append((basis, int(excess.max()), layers[int(np.argmax(excess))], int(ts.sum()),
                             int(rs.max(0).sum()), int(tl.sum()), int(rl.max(0).sum())))
        lines.append("")

    lines += ["## Pairs among loose ratio-type units, trained vs random (all layers)", "",
              "| Pair | Trained | " + " | ".join(randoms) + " |", "|---|---|" + "---|" * len(randoms)]
    counts = {m: Counter() for m in saved}
    for m, s in saved.items():
        for l in layers:
            for basis in ("neurons", "pcs"):
                counts[m].update(s[f"L{l}_{basis}_best"][s[f"L{l}_{basis}_loose"]])
    for pair, n in counts["trained"].most_common(8):
        lines.append(f"| {pair[3:-1]} | {n} | " + " | ".join(str(counts[m][pair]) for m in randoms) + " |")
    plot_pairs(out, counts, randoms, meta["variant"])

    lines += ["", "## Stage 0 decision (provisional rule, written before the data)", ""]
    strong = any(ex >= 3 and ts_sum > 2 * max(rs_sum, 1) for _, ex, _, ts_sum, rs_sum, _, _ in verdict_bits)
    nothing = all(ts_sum == 0 for _, _, _, ts_sum, _, _, _ in verdict_bits)
    for basis, ex, at, ts_sum, rs_sum, tl_sum, rl_sum in verdict_bits:
        lines.append(f"- {basis}: strict units trained {ts_sum} vs random max {rs_sum} (best excess {ex} at layer {at}); "
                     f"loose {tl_sum} vs {rl_sum}.")
    lines += ["", "**Strong go**" if strong else ("**Soft go** (nothing in raw units)" if nothing else "**Needs judgement**"), ""]

    ndvi = []
    for l in layers:
        s = saved["trained"]
        b, d, r = s[f"L{l}_neurons_best"], s[f"L{l}_neurons_direction"], s[f"L{l}_neurons_residual"]
        m = b == "ND(nir,red)"
        if m.any():
            i = int(np.argmax(np.where(m, d, -1)))
            ndvi.append(f"L{l} #{i} (dir {d[i]:.2f}, resid {r[i]:.2f}, {int(m.sum())} with NDVI as best class)")
    lines += ["## Best NDVI-class neuron per layer", "", "; ".join(ndvi) or "none", ""]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    plot(out, table, layers, randoms, meta["variant"])


def plot_pairs(out, counts, randoms, variant, top=8):
    """Bar chart and CSV: which band pairs the loose ratio-type units read, trained vs random-init."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [p for p, _ in counts["trained"].most_common(top)]
    for m in randoms:  # also show pairs common in random nets but rare in the trained one
        pairs += [p for p, _ in counts[m].most_common(3) if p not in pairs]
    trained = np.array([counts["trained"][p] for p in pairs])
    rnd = np.array([[counts[m][p] for p in pairs] for m in randoms])
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "trained", *randoms])
        for i, p in enumerate(pairs):
            w.writerow([p[3:-1], trained[i], *rnd[:, i]])
    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, trained, 0.4, color="#A62B55", label="trained")
    ax.bar(x + 0.2, rnd.mean(0), 0.4, color="0.65", label=f"random-init (mean, range of {len(randoms)})",
           yerr=[rnd.mean(0) - rnd.min(0), rnd.max(0) - rnd.mean(0)], capsize=3)
    ax.set_xticks(x, [p[3:-1].replace(",", "–") for p in pairs], rotation=30, ha="right")
    ax.set_ylabel("loose ratio-type units\n(all layers, neurons + PCs)")
    ax.set_title(f"{variant}: which band pairs the ratio-type units read")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pairs.png", dpi=150)
    plt.close(fig)


def plot(out, table, layers, randoms, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    get = lambda m, b, k: [r[k] for r in table if r["model"] == m and r["basis"] == b]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, basis in zip(axes, ("neurons", "pcs")):
        for i, m in enumerate(randoms):
            ax.plot(layers, get(m, basis, "ratio_loose"), color="0.6", lw=1, label="random-init (loose)" if i == 0 else None)
        ax.plot(layers, get("trained", basis, "ratio_loose"), color="#A62B55", lw=2, label="trained (loose)")
        ax.plot(layers, get("trained", basis, "ratio_strict"), color="#A62B55", lw=2, ls="--", label="trained (strict)")
        ax.set_title(f"{variant}: ratio-type {basis} per layer")
        ax.set_xlabel("layer")
        ax.set_ylabel("units")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "counts.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tiny-tl")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--tokens-per-tile", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--eps", type=float, default=None,
                    help="v3 F1: perturbation size for the census (default 0.05); the output folder "
                         "gets an _eps<value> suffix so a sweep does not overwrite the primary run")
    args = ap.parse_args()
    global EPS
    if args.eps:
        EPS = args.eps
    args.out = args.out or RUNS_DIR / ("step13_pilot" if args.variant == "tiny-tl" else f"step13_pilot_{args.variant}")
    if args.eps:  # keep sweep runs in their own folders
        args.out = args.out.with_name(args.out.name + f"_eps{args.eps}")
    report(args.out) if args.report_only else run(args)


if __name__ == "__main__":
    main()
