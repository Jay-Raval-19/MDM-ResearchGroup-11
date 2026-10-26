# Rediscovering NDVI

Do Earth-observation foundation models compute spectral indices, or only correlate with them?
This repo tests features inside Prithvi-EO-2.0 with spectral interventions whose effects are
exactly predictable for known indices. The full research plan is in `../rediscovering-ndvi.html`.

## Layout

```
src/rndvi/     bands.py     band order, edits, mixing, Prithvi normalization
               indices.py   NDVI, NDWI, NDMI, NBR, SAVI, EVI + near-miss decoys
               spectra.py   illustrative endmembers, synthetic mixed-pixel patches
               battery.py   elasticities, invariance tests, class alignment, verdicts
               config.py    paths and PROVISIONAL thresholds
scripts/       metric_check.py   Stage 0: does the battery classify known formulas?
tests/         run_tests.py      unit tests (no pytest needed)
notes/         notebook.md, prereg/ (frozen thresholds and predictions go here)
runs/          one folder per experiment
```

Large files (HLS tiles, checkpoints, activations) live in `RNDVI_DATA`, which defaults to
`~/rndvi-data`, outside OneDrive.

## Run

```bash
python tests/run_tests.py
python scripts/metric_check.py
```

## Conventions

- Reflectance is HLS surface reflectance x 10,000, band order Blue, Green, Red, NIR, SWIR1, SWIR2.
- Every spectral edit happens in reflectance space, before Prithvi's normalization.
- Thresholds in `config.py` are provisional until frozen in `notes/prereg/thresholds.md` (Stage 0.5).
