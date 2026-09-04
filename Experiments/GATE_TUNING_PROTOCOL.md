# Detector-specific β–θ fairness experiment

This experiment directly addresses the concern that applying the same anomaly-
score smoothing parameter β and decision threshold θ to every detector may
favour some residual distributions over others.

## Confirmatory design

- Use the canonical 60% train / 20% validation / 20% test partition for both
  `firefox-android` and `mozilla-beta`.
- Hold detector hyperparameters at the values already selected for the current
  manuscript: ARIMA(1,1,1); dataset-specific SMA window, EWMA alpha, and compact
  Transformer window/embedding dimension.
- For every detector and dataset, evaluate β in
  `{0.10, 0.20, 0.30, 0.50, 0.70}` and θ in
  `{0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}` on validation data only.
- Select the pair with maximum mean per-signature validation F1. Resolve exact
  ties by choosing the pair closest to the shared default `(0.30, 0.30)`.
- For Transformer, average validation results across the five canonical seeds
  before selection; use one selected pair for the detector, not one pair per
  seed.
- Evaluate the selected pair once on the held-out test segment.
- Compare the resulting detector order with the order produced by the shared
  default `(0.30, 0.30)` using rank order, Spearman rho, and Kendall tau.

The detector prediction/residual sequence and dynamic-threshold anomaly flag do
not depend on β or θ. The implementation therefore generates that sequence once
per signature/model/seed and sweeps the gate parameters afterward. This is
mathematically equivalent to rerunning the gate for each grid point and avoids
35 redundant model fits.

## Outputs

`Experiments/results/gate_tuning/` contains:

- `validation_grid.csv`: all validation results and selection indicators
- `test_grid_summary.csv`: complete test-grid results for transparency; these
  values are not used for parameter selection
- `selected_test_detail.csv`: per-signature held-out results for selected pairs
- `ranking_comparison.csv`: canonical shared-default versus detector-specific
  tuned results and ranks
- `summary.json`: selected parameters, metrics, and rank-stability statistics
- `run_metadata.json`: package versions, input hashes, grids, and elapsed time
- validation heatmaps for every detector and dataset

Run from the repository root:

```bash
python Experiments/detector_specific_gate_tuning.py
```

Checkpoints under `Experiments/results/gate_tuning/cache/` allow interrupted
runs to resume and are excluded from version control.

The fixed input snapshot is available from Zenodo record 14927532 (DOI
`10.5281/zenodo.14927532`). The expected `data.zip` MD5 is
`a6f6d41f262dbb678cdb469d3d94aa13`.
