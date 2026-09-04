# Result-file guide

Use the **canonical** group for the manuscript's primary claims.

## Canonical: 60/20/20, two datasets

- `combined_summary.json`: aggregate metrics and validation-selected parameters
- `combined_detail.csv`: per-signature metrics; deterministic methods have one
  row per signature and Transformer has five seed rows per signature
- `firefox-android/` and `mozilla-beta/`: dataset-specific versions of the
  combined outputs
- `plots/combined_comparison.png`: canonical cross-dataset comparison
- `compute_canonical_stats.py`: bootstrap confidence intervals and paired
  Wilcoxon tests used for the canonical classical-detector comparisons
- `../Experiments/results/transformer_statistics/`: paired Transformer-versus-
  SMA/EWMA tests for the canonical protocol, plus a separately labeled
  detector-specific-gate robustness analysis
- `ewma_extended_tuning.json`: canonical validation-grid extension below
  alpha 0.05; the selected value remains 0.05 on both datasets

## Calibration/supplementary: 70/30, firefox-android

- `summary.json` and `mozilla_results*.csv`: calibration detector outputs
- `stats_significance.json`: calibration statistics, not canonical statistics
- `ablation_*.csv`: calibration sensitivity analyses
- `replay/`: replay/storage experiment
- `fp_adjudication_sample.csv` and `fp_audit_summary.csv`: automated heuristic
  audit of unmatched detections
- method-specific plots named by signature ID: calibration visualizations

Despite the label `mozilla_results`, these files contain the
`firefox-android` calibration subset. Do not infer the dataset from the
filename.

## Exploratory, not reported as canonical evidence

- `bugzilla_validation*`
- `dual_operating_point*`
- `plots/bugzilla_gt_comparison.png`
- `plots/beta_theta_f1_heatmap.png`
- `plots/operating_point_pr_curve.png`

Do not aggregate across these groups. They use different partitions,
hyperparameters, signature subsets, or ground-truth definitions.
