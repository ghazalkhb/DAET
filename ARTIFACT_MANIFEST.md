# Artifact manifest for the manuscript

This repository contains two evaluation protocols. The manuscript reports a
small 70/30 calibration baseline and a larger 60/20/20 canonical validation.
They answer different questions and their output files are not interchangeable.

## Paper-to-artifact map

| Manuscript content | Protocol | Generating script | Frozen output |
|---|---|---|---|
| Dataset scale and evaluation runs | 60/20/20 canonical | `Experiments/data_scale_audit.py` | `Experiments/results/data_scale_audit.json` |
| Main detector comparison, both datasets | 60/20/20 canonical | `Code/run_full_evaluation.py` | `results/combined_summary.json`, `results/combined_detail.csv` |
| Dataset-specific detector results | 60/20/20 canonical | `Code/run_full_evaluation.py` | `results/firefox-android/`, `results/mozilla-beta/` |
| Bootstrap CIs and paired Wilcoxon tests | 60/20/20 canonical | `results/compute_canonical_stats.py` | manuscript tables (script prints values) |
| Transformer and detector efficiency | canonical hyperparameters | `Experiments/transformer_efficiency.py` | `Experiments/results/transformer_efficiency_*.{json,csv}` |
| Initial/calibration detector comparison | 70/30 calibration | detector pipeline | `results/summary.json`, `results/mozilla_results*.csv` |
| Extended EWMA validation grid | 60/20/20 canonical | `Code/15_ewma_extended_tuning.py` | `results/ewma_extended_tuning.json` |
| Detector-specific β–θ fairness check | 60/20/20 canonical robustness | `Experiments/detector_specific_gate_tuning.py` | `Experiments/results/gate_tuning/` |
| Transformer statistical comparisons | canonical + tuned-gate robustness | `Experiments/transformer_statistical_comparison.py` | `Experiments/results/transformer_statistics/` (see `RESULTS.md`) |
| Calibration sensitivity and ARIMA orders | 70/30 calibration | `Code/10_ablation.py`, `Code/16_arima_order_expansion.py` | `results/ablation_*.csv` |
| End-to-end replay | 70/30 calibration | `Code/11_replay_experiment.py` | `results/replay/` |
| FP-label audit | 70/30 calibration | `Code/17_fp_label_audit.py` | `results/fp_adjudication_sample.csv`, `results/fp_audit_summary.csv` |

## Canonical experiment identity

The canonical output is identified by all of the following:

- 342 `firefox-android` signatures and 1,477 `mozilla-beta` signatures
- 214,063 and 1,183,738 telemetry points, respectively
- 60/20/20 chronological partitions
- 1,710 and 7,385 detector-signature pairs before expansion of Transformer
  seeds (9,095 total)
- four deterministic methods plus five Transformer seeds per signature
- tuned parameters stored in `results/combined_summary.json`

Because `combined_detail.csv` stores all five Transformer seeds, its physical
row count is 16,371 rather than 9,095. The latter is the conceptual count of
five method families × signatures reported by the manuscript.

## Non-canonical files

`results/summary.json`, `results/mozilla_results*.csv`, and
`results/stats_significance.json` belong to the 70/30 calibration
experiment. Their larger calibration F1 values must not be presented as the
two-dataset canonical results.

`Code/13_bugzilla_validation.py`, `Code/14_dual_operating_point.py`,
`results/bugzilla_validation*`, and `results/dual_operating_point*` are retained
as exploratory analyses. They are not sources for a table or claim in the
manuscript.

Order-selection experiments that did not use a comparable held-out protocol
are intentionally excluded from the manuscript artifact.

## Reproduction boundary

The frozen results are included, but the large raw `Data/timeseries-data/`
directory is not. Complete reruns require reacquiring those per-signature CSVs.
Some calibration scripts additionally refer to intermediate vectors
that are not included; their frozen output is retained for traceability. The
canonical pipeline is self-contained once the raw time-series directory is
provided.
