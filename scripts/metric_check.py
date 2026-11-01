"""Stage 0 metric check: does the battery classify known formulas correctly?

Runs every near-miss function through the full battery on synthetic mixed-pixel
patches, then sweeps the NDVI leak to measure the method's detection limit.
Writes runs/metric_check/report.md and results.csv.

    python scripts/metric_check.py
"""
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rndvi import indices as ix  # noqa: E402
from rndvi.battery import alignment_cosines, elasticities, ratio_residual, score  # noqa: E402
from rndvi.config import RUNS_DIR, THRESHOLDS  # noqa: E402
from rndvi.spectra import synthetic_patches  # noqa: E402

N_PATCHES, PATCH_SIZE, SEED = 2000, 4, 0
LEAKS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def passes(verdict: str, expected: str, best: str) -> bool:
    if expected.startswith("ratio:"):
        return verdict.startswith(f"ratio-type {expected[6:]}")
    if expected == "not-ratio":
        return not verdict.startswith("ratio-type")
    return best == expected and not verdict.endswith("weak")


def main():
    out_dir = RUNS_DIR / "metric_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = synthetic_patches(N_PATCHES, PATCH_SIZE, SEED)

    rows = []
    for name, (f, expected) in ix.near_miss_suite().items():
        res = score(f, x, THRESHOLDS)[0]
        ok = passes(res["verdict"], expected, res["best"])
        rows.append({"function": name, "expected": expected, "pass": ok, **res})

    # Detection limit: how big an NIR leak does the ratio residual catch?
    leak_rows = []
    for eps in LEAKS:
        f = ix.patch_mean(ix.leaky_ndvi(eps))
        e = elasticities(f, x)
        names, cos = alignment_cosines(e, x)
        leak_rows.append({
            "eps": eps,
            "ratio_residual": float(ratio_residual(e)[0]),
            "cos_to_NDVI_template": float(np.median(cos[:, 0, names.index("ND(nir,red)")])),
        })
    caught = [r["eps"] for r in leak_rows if r["ratio_residual"] >= THRESHOLDS["ratio_residual"]]
    detection_limit = min(caught) if caught else None

    # Aggregation: NDVI of the mean spectrum vs mean of per-pixel NDVI on mixed patches
    agg_gap = np.abs(ix.patch_mean(ix.ndvi)(x) - ix.index_of_mean(ix.ndvi)(x))

    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(r["pass"] for r in rows)
    lines = [
        f"# Stage 0 metric check · {date.today().isoformat()}",
        "",
        f"{N_PATCHES} synthetic mixed-pixel patches ({PATCH_SIZE}x{PATCH_SIZE}, seed {SEED}); "
        f"provisional thresholds {THRESHOLDS}.",
        "",
        f"**{n_pass}/{len(rows)} functions classified correctly.**",
        "",
        "| Function | Expected | Verdict | Direction | Margin [95% CI] | Ratio residual | "
        "All-band S | Own-pair S | Pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["margin_ci"]
        lines.append(
            f"| {r['function']} | {r['expected']} | {r['verdict']} | {r['direction']:.3f} | "
            f"{r['margin']:.3f} [{lo:.3f}, {hi:.3f}] | {r['ratio_residual']:.3f} | "
            f"{r['all_band_S']:.3f} | {r['own_pair_S']:.3f} | {'yes' if r['pass'] else 'NO'} |")
    lines += [
        "",
        "## Detection limit for an NIR leak (NDVI + eps*NIR)",
        "",
        "| eps | Ratio residual | Cosine to NDVI template |",
        "|---|---|---|",
    ]
    lines += [f"| {r['eps']} | {r['ratio_residual']:.4f} | {r['cos_to_NDVI_template']:.4f} |"
              for r in leak_rows]
    lines += [
        "",
        f"Smallest leak caught at ratio-residual threshold {THRESHOLDS['ratio_residual']}: "
        f"**eps = {detection_limit}**.",
        "",
        "## Aggregation on mixed patches",
        "",
        f"|mean(pixel NDVI) - NDVI(mean spectrum)|: median {np.median(agg_gap):.4f}, "
        f"max {agg_gap.max():.4f}. Both are expected in the same ratio class.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
