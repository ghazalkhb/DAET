# Comparing Change Detectors for Adaptive Execution Tracing

This is the replication repository for **“Comparing Change Detectors for
Adaptive Execution Tracing: Large-Scale Validation and Design Insights.”** It
compares ARIMA, LAST, SMA, EWMA, and a compact Transformer on Mozilla
Treeherder performance time series.

The main evaluation uses the two-dataset, chronological **60% train / 20%
validation / 20% test** protocol in `Code/run_full_evaluation.py`. The separate
70/30 outputs implement the calibration baseline reported in the manuscript;
they are not the canonical evaluation.

## Canonical protocol

- Datasets: `firefox-android` and `mozilla-beta`
- Proxy ground truth: Treeherder alert timestamps
- Split: chronological 60/20/20 for every detector
- Validation use: selection of SMA window, EWMA alpha, and Transformer window
  and embedding dimension
- Fixed detector gate: beta = 0.30, anomaly threshold = 0.30, 3-sigma
  thresholding, threshold-history window = 30, and alert tolerance = ±5 samples
- Transformer: one encoder layer, two attention heads, MSE loss, Adam, early
  stopping, and five seeds (`42, 123, 456, 789, 1024`)
- Metrics: precision, recall, F1, false-alert rate (FAR), detection rate, delay,
  and storage reduction

The selected hyperparameters are:

| Dataset | SMA | EWMA | Transformer |
|---|---:|---:|---|
| firefox-android | window 40 | alpha 0.05 | window 20, d_model 16 |
| mozilla-beta | window 20 | alpha 0.05 | window 10, d_model 32 |

## Canonical results

These values come from `results/combined_summary.json` and correspond to the
manuscript's main result tables.

### firefox-android

| Method | Precision | Recall | F1 | FAR | Storage reduction | Detection rate |
|---|---:|---:|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.0560 | 0.2573 | 0.0893 | 0.056253 | 0.9030 | 66.18% |
| LAST | 0.0455 | 0.2091 | 0.0718 | 0.054508 | 0.8915 | 54.41% |
| SMA(w=40) | 0.0661 | 0.2895 | 0.1039 | 0.054695 | 0.8826 | 72.79% |
| EWMA(a=0.05) | 0.0676 | 0.2632 | 0.1033 | 0.052155 | 0.8943 | 67.65% |
| Transformer(w=20,d=16) | 0.0688 | 0.2924 | 0.1051 | 0.065694 | 0.6721 | 74.41% |

### mozilla-beta

| Method | Precision | Recall | F1 | FAR | Storage reduction | Detection rate |
|---|---:|---:|---:|---:|---:|---:|
| ARIMA(1,1,1) | 0.0503 | 0.2317 | 0.0761 | 0.060171 | 0.9031 | 88.61% |
| LAST | 0.0454 | 0.2115 | 0.0685 | 0.053469 | 0.8903 | 81.77% |
| SMA(w=20) | 0.0574 | 0.2408 | 0.0847 | 0.059089 | 0.8884 | 91.65% |
| EWMA(a=0.05) | 0.0570 | 0.2363 | 0.0830 | 0.059159 | 0.8847 | 90.63% |
| Transformer(w=10,d=32) | 0.0539 | 0.2348 | 0.0789 | 0.064139 | 0.7109 | 89.82% |

## Repository layout

```text
Code/
  run_full_evaluation.py       canonical two-dataset evaluation
  10_ablation.py               calibration sensitivity analyses
  11_replay_experiment.py      supplementary replay experiment
  12_fp_adjudication.py        false-positive sample construction
  15_ewma_extended_tuning.py   EWMA sensitivity analysis
  16_arima_order_expansion.py  ARIMA order sensitivity analysis
  17_fp_label_audit.py         heuristic FP-label audit
  13_bugzilla_validation.py    exploratory, not a canonical result
  14_dual_operating_point.py   exploratory, not a canonical result
Experiments/
  transformer_efficiency.py Transformer/runtime efficiency experiment
  data_scale_audit.py       dataset-size and run-count audit
  detector_specific_gate_tuning.py detector-specific β–θ fairness test
  transformer_statistical_comparison.py paired Transformer statistics
Data/
  alerts_data.csv              Treeherder alert metadata
  bugs_data.csv                linked Bugzilla records
  scripts/                     data collection and preprocessing utilities
results/
  combined_summary.json        canonical aggregate results
  combined_detail.csv          canonical per-signature results
  firefox-android/             canonical dataset-specific outputs
  mozilla-beta/                canonical dataset-specific outputs
  replay/                      supplementary replay outputs
  ...                          calibration and exploratory outputs (mapped below)
ARTIFACT_MANIFEST.md           paper-to-artifact map and protocol boundaries
verify_artifact.py             fast consistency check for included outputs
```

See `results/README.md` before using any result file. It separates canonical
results from calibration and exploratory analyses so that different protocols
are not accidentally combined.

## Data availability

`Data/alerts_data.csv` and `Data/bugs_data.csv` are included. The raw
per-signature time-series files are not included because of their size. To
rerun the canonical pipeline, place them at:

```text
Data/timeseries-data/firefox-android/<signature_id>_timeseries_data.csv
Data/timeseries-data/mozilla-beta/<signature_id>_timeseries_data.csv
```

The collection utilities in `Data/scripts/` use Mozilla's public Treeherder and
Bugzilla APIs. See `Data/README.md` for the schema and acquisition notes. The
precomputed canonical outputs remain inspectable and verifiable without the raw
time-series files.

## Environment and verification

Create an isolated environment and install the manuscript-reported dependency
set:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_artifact.py
```

`verify_artifact.py` checks result schemas, row/signature counts, selected
hyperparameters, method/seed coverage, dataset-scale totals, and consistency
between the dataset-specific and combined outputs. Missing raw time-series data
is reported as an informational limitation, not as corruption of the included
precomputed artifact.

## Reproduction

With the raw time series installed in the paths above:

```bash
# Canonical 60/20/20 evaluation (the manuscript's main tables)
python Code/run_full_evaluation.py

# Canonical statistical analysis, run from the results directory
(cd results && python compute_canonical_stats.py)

# Efficiency and data-scale analyses
python Experiments/transformer_efficiency.py
python Experiments/data_scale_audit.py

# Detector-specific β–θ fairness/robustness experiment
python Experiments/detector_specific_gate_tuning.py

# Paired Transformer comparisons against SMA and EWMA
python Experiments/transformer_statistical_comparison.py

# Canonical extended EWMA validation grid
python Code/15_ewma_extended_tuning.py
```

The ablation and replay scripts reproduce the manuscript's explicitly labeled
calibration/supplementary analyses. They do not implement the canonical
two-dataset protocol and should not overwrite or replace `combined_*.{csv,json}`.

## Reproducibility boundary

The included CSV and JSON files are the frozen outputs used by the current
manuscript. Rerunning the complete experiment requires the omitted time-series
directory and substantial compute, particularly for ARIMA and the five-seed
Transformer. The manuscript source itself is intentionally not part of this
corrected repository copy and has not been modified.
