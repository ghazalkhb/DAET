# DAET: Empirical Evaluation of Anomaly Detection on Mozilla Performance Data

This repository contains the code and results for an empirical study comparing anomaly detection methods on real-world software performance time-series data from Mozilla's CI pipeline (Treeherder).

The study evaluates how well classical statistical detectors — and a Transformer-based approach — agree with Treeherder's alert system, using precision, recall, F1, false-alert rate (FAR), detection rate, and storage reduction as metrics.

---

## Methods Evaluated

| Method | Type |
|---|---|
| LAST | Baseline (last-value threshold) |
| SMA | Simple Moving Average |
| EWMA | Exponentially Weighted Moving Average |
| ARIMA | AutoRegressive Integrated Moving Average |
| Transformer (DAET) | Deep learning (Huber forecasting + anomaly head + GRAC) |

---

## Key Results (firefox-android, canonical evaluation)

| Method | Precision | Recall | F1 | FAR | Det. Rate |
|---|---|---|---|---|---|
| LAST | 0.079 | 0.484 | 0.133 | 0.052 | 72.67% |
| SMA | 0.112 | 0.657 | 0.186 | 0.052 | 96.00% |
| EWMA | 0.090 | 0.562 | 0.151 | 0.056 | 82.67% |
| ARIMA | 0.106 | 0.606 | 0.174 | 0.053 | 88.67% |
| Transformer | 0.208 | 0.334 | 0.229 | 0.013 | 48.00% |

Evaluation metric: agreement with Treeherder alert labels (proxy ground truth).

---

## Repository Structure

```
DAET/
├── Code/                        # All evaluation scripts
│   ├── run_full_evaluation.py   # Main pipeline runner
│   ├── helpers.py               # Shared utilities
│   ├── 10_ablation.py           # Ablation study (β, τ, SMA window, threshold)
│   ├── 11_replay_experiment.py  # Replay / storage reduction experiment
│   ├── 12_fp_adjudication.py    # False-positive audit sampling
│   ├── 13_bugzilla_validation.py# Cross-validation against Bugzilla bug reports
│   ├── 14_dual_operating_point.py  # β×θ grid search (precision-recall trade-off)
│   ├── 15_ewma_extended_tuning.py  # Extended EWMA α tuning
│   ├── 16_arima_order_expansion.py # ARIMA order (p,d,q) ablation
│   ├── 17_fp_label_audit.py     # FP label audit
│   ├── generate_report.py       # Result aggregation and report generation
│   └── copy_plots.py            # Plot organisation helper
│
├── Data/                        # Dataset files
│   ├── alerts_data.csv          # Treeherder alert metadata
│   ├── bugs_data.csv            # Bugzilla bug records
│   ├── README.md                # Dataset description
│   └── scripts/                 # Data extraction and preprocessing scripts
│
├── results/                     # All output files: CSVs, JSONs, plots, logs
│   ├── combined_summary.json    # Aggregated evaluation summary
│   ├── combined_detail.csv      # Per-signature detailed results
│   ├── ablation_*.csv           # Ablation study results
│   ├── bugzilla_validation.*    # Bugzilla cross-validation results
│   └── ...
│
└── ARIMA.ipynb                  # Exploratory notebook for ARIMA analysis
```

---

## Data

The dataset is based on Mozilla's publicly available performance testing infrastructure (Treeherder).

- **`alerts_data.csv`** — alert records from Treeherder (signature ID, push time, alert status, etc.)
- **`bugs_data.csv`** — associated Bugzilla bug reports
- **`timeseries-data/`** — raw per-signature performance time-series (11k+ CSV files, not included in this repo due to size)

The full timeseries data can be obtained from the Mozilla Treeherder public API or the original dataset release. See `Data/README.md` for the dataset schema and structure.

---

## Running the Evaluation

### Requirements

```bash
pip install -r Data/scripts/requirements.txt
```

Core dependencies: `pandas`, `numpy`, `statsmodels`, `matplotlib`, `scikit-learn`, `scipy`

### Full evaluation pipeline

```bash
python Code/run_full_evaluation.py
```

This runs all detectors on all signatures and writes outputs to `results/`.

### Individual analyses

```bash
# Ablation study
python Code/10_ablation.py

# Replay / storage reduction experiment
python Code/11_replay_experiment.py

# False-positive audit
python Code/12_fp_adjudication.py

# Dual operating point (β × θ grid)
python Code/14_dual_operating_point.py

# Extended EWMA tuning
python Code/15_ewma_extended_tuning.py
```

---

## Results

Pre-computed results are included in the `results/` directory:

- `combined_summary.json` / `combined_detail.csv` — main evaluation results across all methods and signatures
- `ablation_beta.csv`, `ablation_tau.csv`, `ablation_sma_window.csv`, `ablation_threshold.csv` — parameter sensitivity
- `ablation_arima_order.csv` — ARIMA order sensitivity
- `bugzilla_validation.csv` / `.json` — Bugzilla cross-validation
- `stats_significance.json` — Wilcoxon signed-rank test results
- `bootstrap_ci.json` — Bootstrap confidence intervals

---

## Evaluation Protocol

- **Ground truth**: Treeherder alert labels (proxy for true anomalies)
- **Train/test split**: 70% training, 30% test (time-ordered)
- **Dataset**: firefox-android and mozilla-beta signature groups
- **Statistical tests**: Wilcoxon signed-rank (pairwise), bootstrap CI (1000 iterations)
