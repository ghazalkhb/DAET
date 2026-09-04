# Detector-specific β–θ tuning results

This report records the completed canonical robustness experiment.

## Protocol

- Fixed May 2023–May 2024 dataset snapshot from Zenodo record 14927532
  (`data.zip` MD5 `a6f6d41f262dbb678cdb469d3d94aa13`)
- 342 Firefox-Android and 1,477 mozilla-beta signatures
- Chronological 60% train / 20% validation / 20% test split
- Existing canonical detector hyperparameters held fixed
- 35 gate configurations per detector and dataset: five β values × seven θ
  values
- Parameters selected only by mean validation F1
- Transformer selection averaged over seeds 42, 123, 456, 789, and 1024
- One selected detector-level pair evaluated on the held-out test segment

## Selected gates and held-out results

| Dataset | Detector | β | θ | Precision | Recall | F1 | FAR | Storage reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Firefox-Android | ARIMA(1,1,1) | 0.20 | 0.25 | 0.0949 | 0.2281 | 0.1266 | 0.0250 | 0.9289 |
| Firefox-Android | LAST | 0.20 | 0.10 | 0.0586 | 0.2442 | 0.0913 | 0.0418 | 0.7035 |
| Firefox-Android | SMA(w=40) | 0.10 | 0.25 | 0.1930 | 0.2719 | 0.2138 | 0.0102 | 0.9347 |
| Firefox-Android | EWMA(a=0.05) | 0.10 | 0.20 | 0.1796 | 0.2719 | 0.2031 | 0.0149 | 0.9121 |
| Firefox-Android | Transformer(w=20,d=16) | 0.10 | 0.25 | 0.1145 | 0.2453 | 0.1462 | 0.0188 | 0.6945 |
| mozilla-beta | ARIMA(1,1,1) | 0.30 | 0.50 | 0.0688 | 0.1161 | 0.0810 | 0.0103 | 0.9842 |
| mozilla-beta | LAST | 0.10 | 0.10 | 0.0549 | 0.2298 | 0.0826 | 0.0406 | 0.6644 |
| mozilla-beta | SMA(w=20) | 0.30 | 0.50 | 0.0882 | 0.1728 | 0.1102 | 0.0127 | 0.9729 |
| mozilla-beta | EWMA(a=0.05) | 0.30 | 0.50 | 0.0878 | 0.1742 | 0.1089 | 0.0137 | 0.9701 |
| mozilla-beta | Transformer(w=10,d=32) | 0.10 | 0.10 | 0.0713 | 0.2474 | 0.1007 | 0.0374 | 0.4626 |

## Ranking robustness

Firefox-Android:

- Shared default: Transformer > SMA > EWMA > ARIMA > LAST
- Detector-specific tuning: SMA > EWMA > Transformer > ARIMA > LAST
- Spearman ρ = 0.70; Kendall τ = 0.60

mozilla-beta:

- Shared default: SMA > EWMA > Transformer > ARIMA > LAST
- Detector-specific tuning: SMA > EWMA > Transformer > LAST > ARIMA
- Spearman ρ = 0.90; Kendall τ = 0.80

The complete rank order is therefore not invariant to gate tuning. The central
practical conclusion is more robust: after independent validation tuning, SMA
ranks first and EWMA second on both datasets, and both remain ahead of the
compact Transformer.

Several selected settings lie at the boundary of the pre-specified reasonable
grid. Results therefore support robustness within this grid; they should not be
described as globally optimal over all possible β and θ values.

## Reproduction check

At the shared `(β=0.30, θ=0.30)` cell, recomputed F1 differs from the frozen
canonical aggregate by at most `0.00004971`, entirely attributable to the
canonical JSON's four-decimal rounding. This confirms that the experiment uses
the same data, detector outputs, split, and metric implementation.
